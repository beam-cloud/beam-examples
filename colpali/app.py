import logging
from beam import endpoint, Image, Volume, env

import pprint
from typing import List, cast
import os

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from colpali_engine.models import ColPali
from colpali_engine.models.paligemma.colpali.processing_colpali import ColPaliProcessor
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor
from colpali_engine.utils.torch_utils import ListDataset, get_torch_device
from PIL import Image as PILImage  # To handle image saving

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the custom image
image = (
    Image()
    .add_commands(["apt-get update -y"])
    .add_python_packages(
        ["colpali_engine==0.3.1", "datasets==3.0.1"]
    )
)

volume_path = "./colpali_volume"  # Path to save images

def on_start():
    device = get_torch_device("auto")

    # Model name
    model_name = "vidore/colpali-v1.2"

    logger.info("Beginning to load model")
    # Load model
    model = ColPali.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    ).eval()
    logger.info("Model has been loaded")
    return model

@endpoint(
    name="colpali-deployment",
    image=image,
    cpu=4,
    memory="32Gi",
    gpu="A10G",
    volumes=[
        Volume(
            name="colpali_volume",
            mount_path=volume_path,
        )
    ],
    on_start=on_start,
)
def main(context):
    """
    Example script to run inference with ColPali and log results.
    """
    # Model name
    model_name = "vidore/colpali-v1.2"
    model = context.on_start_value

    # Load processor
    processor = cast(ColPaliProcessor, ColPaliProcessor.from_pretrained(model_name))

    if not isinstance(processor, BaseVisualRetrieverProcessor):
        raise ValueError("Processor should be a BaseVisualRetrieverProcessor")

    # NOTE: Only the first 16 images are used for demonstration purposes
    dataset = cast(Dataset, load_dataset("vidore/docvqa_test_subsampled", split="test[:16]"))
    images = dataset["image"]

    # Select a few queries for demonstration purposes
    query_indices = [12, 15]
    queries = [dataset[idx]["query"] for idx in query_indices]
    logger.info("Selected queries:")
    logger.info(dict(zip(query_indices, queries)))

    # Run inference - docs
    dataloader = DataLoader(
        dataset=ListDataset[str](images),
        batch_size=4,
        shuffle=False,
        collate_fn=lambda x: processor.process_images(x),
    )
    ds: List[torch.Tensor] = []
    for batch_doc in tqdm(dataloader):
        with torch.no_grad():
            # Ensure tensor indices are in long type if required
            batch_doc = {k: v.to(model.device).to(torch.bfloat16 if k != 'input_ids' else torch.long) for k, v in batch_doc.items()}
            embeddings_doc = model(**batch_doc)
        ds.extend(list(torch.unbind(embeddings_doc.to("cpu"))))

    # Run inference - queries
    dataloader = DataLoader(
        dataset=ListDataset[str](queries),
        batch_size=4,
        shuffle=False,
        collate_fn=lambda x: processor.process_queries(x),
    )

    qs: List[torch.Tensor] = []
    for batch_query in dataloader:
        with torch.no_grad():
            # Ensure tensor indices are in long type if required
            batch_query = {k: v.to(model.device).to(torch.bfloat16 if k != 'input_ids' else torch.long) for k, v in batch_query.items()}
            embeddings_query = model(**batch_query)
        qs.extend(list(torch.unbind(embeddings_query.to("cpu"))))

    # Run scoring
    scores = processor.score(qs, ds).cpu().numpy()
    idx_top_1 = scores.argmax(axis=1)

    # Log detailed results for each query
    for i, query_idx in enumerate(query_indices):
        logger.info(f"\nQuery {query_idx}: {queries[i]}")
        logger.info(f"Top-1 retrieved document index: {idx_top_1[i]}")
        logger.info("Score: %s", scores[i, idx_top_1[i]])

        # Retrieve the document content (image)
        image = dataset[int(idx_top_1[i])]["image"]
        logger.info("Document content: %s", image)

        # Save the image to the volume
        if isinstance(image, PILImage.Image):  # Ensure it's a PIL Image
            output_path = os.path.join(volume_path, f"retrieved_image_{query_idx + 100}.png")
            image.save(output_path)
            logger.info(f"Image saved to {output_path}")
        else:
            logger.warning("Image not in the correct format and cannot be saved.")

    return "success"


if __name__ == "__main__":
    main()
