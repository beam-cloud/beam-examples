const express = require("express");
const beam = require("@beamcloud/beam-js");
const app = express();
const port = 3333;

app.get("/deployment", async (req, res) => {
  const client = await beam.init(process.env.BEAM_TOKEN);

  const dep = await client.deployments.list();

  const callResp = await dep[0].call();
  res.send(callResp.data);
});

app.listen(port, () => {
  console.log(`Example app listening on port ${port}`);
});