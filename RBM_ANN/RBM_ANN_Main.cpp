#define _CRT_SECURE_NO_WARNINGS
#include "RBM.h"
#include "ANN.h"
#include <iostream>
using namespace std;

typedef struct {
    string file; //训练集文件
    int out; //输出层个数，即有2^out种输出
    int num; //训练集个数
} TrainTxt;

TrainTxt tt[] = {
    { "./train_test/train01.txt", 1, 300},  //[0]
    { "./train_test/train0123.txt", 2, 600},   //[1]
    { "./train_test/train012345.txt", 3, 900},    //[2]
    { "./train_test/train01234567.txt", 3, 1200},   //[3]
    { "./train_test/train0123456789.txt", 4, 1500},   //[4]
};

int main()
{
    cout << RBM::getDateTime() << endl;
    SetText(FG_HL | FG_G | FG_B);
    try {
        TrainTxt &t = tt[2]; //通过修改序号载入不同的训练集
        int hideUnits[] = { 144, 36 };
        RBM rbm(784, hideUnits);
        cout << "\r>>>正在载入训练数据和测试数据...";
        time_t t_start = clock();
        rbm.loadTrain(t.file, t.num);
        cout << "\r载入训练和测试数据耗时: " << (clock() - t_start) << "ms" << endl;
        t_start = clock();
        rbm.train(0.05, 10000);  //允许误差和最大代数，任意一个满足则停止
        cout << "演化共耗时: " << (clock() - t_start) / 1000.0 << " s\n\n";
        rbm.saveRBMOutToFile("rbmOut.txt"); //将RBM的输出层以及标签保存到文件
        uint rbmOutSize = hideUnits[sizeof(hideUnits) / sizeof(hideUnits[0]) - 1];
        ///将RBM抽取的特征送到ANN中进行分类
        ANN ann(rbmOutSize, t.out, 1); //输入数据大小，输出层神经元个数，隐层层数
        ann.loadTrainSet("rbmOut.txt"); //读取RBM的输出作为ANN的训练数据
        ann.train(0.02, 1000000);
    } catch (const logic_error &err) {
        cout << "\r---error:" << err.what() << endl;
    } catch (...) {
        cout << "\nOops, there are some jokes in the runtime, I am lost in the jungle\\(╯-╰)/" << endl;
    }
    cin.get();
    return 0;
}
