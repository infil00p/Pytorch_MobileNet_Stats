# Pytorch_MobileNet_Stats

Application for comparing numerous ML frameworks on the Mobilnet model.  This app does the folowing:
- Runs MobileNet on 25 images located in assets
- Generates JSON reports on each of inference results based on the URI, saves to shared storage on the device
- Plots a graph in seconds on the screen (Makes for a cool demo, not super useful)

This currently works on CPU and NNAPI, but breaks on GPU due to too many ML frameworks trying to use the GPU at once,  This was a cool demo idea, but things break unexpectedly.

As with any work-related demo that's not strictly work, it's covered under the Apache Licence 2.0
