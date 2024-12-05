# Load local checkpoint and model weights on Beam

This example demonstrates how to load from local a checkpoint, lora, vae and ti for base model Stable Diffusion v1.5

Credits to [Talbo](https://x.com/TalboSocial)

## Overview

This app has an APIs to generate an image based on the prompt.

# Pre-requisites 

1. Make sure you have [Beam](https://beam.cloud) installed: `curl https://raw.githubusercontent.com/slai-labs/get-beam/main/get-beam.sh -sSfL | sh`
2. Clone this repo and `cd` into the directory

# Quickstart

0. Go to your Beam dashboard and upload the model weights on a Volume you create as described [here](https://docs.beam.cloud/data/volumes#uploading-files-with-the-dashboard).
Then make sure you add the path to each model weight in the constants section:
```
model_path = f"{volume_path}/Anything-V3.0-pruned-fp32.safetensors"
lora_path = f"{volume_path}/"
lora_name = "Crayon.safetensors"
ti_path = f"{volume_path}/1vanakn0ll.pt"
vae_path = f"{volume_path}/vae-ft-mse-840000-ema-pruned.safetensors"
```

If you can not load a weight from a single file then you must do the following steps:

  a. download the entire folder of the weight model from HuggingFace and upload it to your beam volume
  b. then use the "from_pretrained" method and pass the path to the folder not the model_id:
    ```pipe.vae = AutoencoderTiny.from_pretrained("./models/vae/your-vae/", torch_dtype=precision_consistency).to("cuda")```

1. Test the API Locally: `beam serve app.py:generate_image`. You can make any desired changes to the code, and Beam will automatically 
  reload the remote server each time you update your application code. 
  Note: Any updates to compute requirements, python packages, or shell commands will require you to manually restart the dev session
2. Deploy the API to Beam: `beam deploy app.py --name sd-load-local-weights`
  Once it's deployed, you can find the web URL in the dashboard.


## Calling the Inference API

Here's what a request will look like:

```curl

curl -X POST \
    --compressed 'https://uc6mc.apps.beam.cloud' \
    -H 'Accept: */*' \
    -H 'Accept-Encoding: gzip, deflate' \
    -H 'Authorization: Basic YOUR_AUTH_KEY' \
    -H 'Connection: keep-alive' \
    -H 'Content-Type: application/json' \
    -d '{"prompt": "photo of a girl riding the subway"}'
```

The main changes for beam v2:
- Don't define an App anymore
- Define an Image and a Volume that you pass to @endpoint as args
- In volume, path is renamed mount_path
- @app.rest_api() is now @endpoint(image=image, cpu=4, memory="16Gi", gpu="T4", volumes=[volume])
- Loader is now called on_start
- Loader context is retrieved via context.on_start_value
- Imports need to be inline with your remote functions
