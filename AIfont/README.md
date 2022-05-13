
# Aifont
# 1. Prepare the dataset
You can use the tools(TTFtoPNG.py) to convert the font file TTF to PNG,or you can collect font images of the target style.

# 2. Train
You can refer to https://github.com/ZC119/Handwritten-CycleGAN during model training.

# 3. Trace
If you want to inference in BM1684,you need trace your model.
Put the trace.py file in the root directory of the training file.
```
python trace.py --dataroot datasets/testdataset/ --cuda --input_nc 1 --output_nc 1 --size 128 --generator_A2B output_1/19_netG_A2B.pth
```
# 4. Model migration
You can find information about model migration in the documentation of the BM1684 BMNNSDK2.
https://sophgo-doc.gitbook.io/bmnnsdk2-bm1684/model-convert/fp32-bmodel/pt
Ps:If you use grayscale image, the shape is [1,1,128,128]. If you use RGB image, the shape is [1,3,128,128].(128 is the size of image)

# 5. Model deployment
Put cyclegan.py, bmodel and standard kaiti TTF file into sophongo's products(such as se5).
```
python3 cyclegan.py
```
Result will be saved in Current directory.

