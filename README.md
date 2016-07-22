# RBM_ANN_Demo

整个项目代码的主入口是RBM_ANN_Main.cpp

1.先将代码所在目录的train_test.7z压缩文件解压到代码所在目录，形成以下目录结构：
```---------------------------
|-说明.txt  
|-RBM_ANN.sln  
|-RBM_ANN  
  |-ANN.cpp  	
  |-ANN.h  
  |-RBM.cpp  
  |-RBM.h  
  |-MathUtil.hpp  
  |-Win_Util.h  
  |-RBM_ANN_Main.cpp  
  |-set.ini  
  |-RE2JPG.exe  
  |-saveRGBToFile.exe  
  |-train_test  
    |-train01.txt  
    |-train0123.txt  
    |-train012345.txt  
    |-train01234567.txt  
    |-train0123456789.txt  
---------------------------
```
2.RBM_Main.cpp中有一个全局结构体变量g_tt用于保存若干可用的数据集信息，
可通过在配置文件set.ini中修改RBM_ANN节点的TrainTxtIdx值改变数据集组序号。

3.可在第55行修改数组，从而调整网络的层数及每层个数

4.可调整第58/65行的两个参数，来改变训练需求/停止条件

5.getImgPredictOut()函数可允许在训练好rbm和ann模型后拖入图像文件到控制台
然后预测图像对应的标签输出出来，如果是做成有UI的应用程序，则可模仿该函数
内的处理逻辑（第31-34行）,具体函数的用法及参数设置参考相应头文件和源文件

6.附带的ANN_Main.cpp和RBM_Main.cpp是用于对ANN和RBM进行单测的主入口，如果
不是为了单测，不需要载入到项目中。

7.其中有两个exe文件，都需要依赖于opencv环境才能正常运行，请至少配置好其
运行环境，如果需要进行opencv开发任务，则还需要配置好编译环境。  
 `:RE2JPG.exe`用于在训练完模型之后将图像正向通过模型一遍，再反向反演回去，
得到原图的重构图，可用来查看训练效果的实际效果  
 `:saveRGBToFile.exe`用于读取图像文件然后把图像的像素（灰度）数据保存至一
个文件中，便于后续读取该图片的像素数据。  

 如果暂未配置好opencv运行环境，也可以将RBM.cpp第433行调用RE2JPG.exe的代码
注释，且也不支持拖入图像文件得到预测结果，但仍可以载入RGB数据文件预测结果

8.部分代码需编译器支持C++11特性，如果实在无法支持C++11,可将RBM_ANN_Main.cpp
[40-41]改为：
```cpp
for(int i=0; i<tags.size(); ++i)
    cout<<tags[i]<<" ";
```
