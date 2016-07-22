#include "RBM.h"
#include "ANN.h"
#include "Math_Util.hpp"
#include <iostream>
using namespace std;

typedef struct {
    string file; //训练集文件路径
    int n_out;   //输出层个数
    int n_train; //训练集个数
} TrainTxt;

TrainTxt tt__[] = {
    { "./train_test/train01.txt", 2, 250},  //[0]
    { "./train_test/train0123.txt", 4, 500},   //[1]
    { "./train_test/train012345.txt", 6, 750},    //[2]
    { "./train_test/train01234567.txt", 8, 1000},   //[3]
    { "./train_test/train0123456789.txt", 10, 1250},   //[4]
};

void getImgPredictOut(RBM& rbm, ANN& ann)
{
    string file;
    while (getline(cin, file)) {
        system(("saveRGBToFile.exe " + file).c_str());
        char saveToFile[255] = "";
        GetPrivateProfileString("saveRGBToFile", "saveToFile", "", saveToFile, 255, ".\\set.ini");
        streambuf* coutBuf = cout.rdbuf();
        ofstream fout("tmp");
        cout.rdbuf(fout.rdbuf());
        rbm.loadTestSet(saveToFile, 0, 0);
        rbm.saveRBMOutToFile("rbm.out", 0);
        ann.loadTestSet("rbm.out", 0, 0);
        vector<int> tags = ann.getTestOut();
        fout.close();
        cout.rdbuf(coutBuf);
        DeleteFile(saveToFile);
        DeleteFile("rbm.out");
        DeleteFile("tmp");
        for (auto& elem : tags)
            cout << elem << " ";
        cout << endl;
    }
}

int main()
{
    int idx = GetPrivateProfileInt("RBM_ANN", "TrainTxtIdx", 0, ".\\set.ini");
    char learnRate[10] = "";
    GetPrivateProfileString("RBM_ANN", "LearnRate", "0.3", learnRate, 10, ".\\set.ini");
    TrainTxt& tt = tt__[idx]; ///通过修改序号载入不同的训练集
    cout << Math_Util::getDateTime() << "\t" << tt.file << "\t" << tt.n_train << endl;
    SetText(FG_HL | FG_G | FG_B);
    try {
        int hideUnits[] = { 196, 49 };
        RBM rbm(784, hideUnits, atof(learnRate));
        rbm.loadTrainSet(tt.file, tt.n_train);
        rbm.train(0.05, 10000);  //允许误差和最大代数，任意一个满足则停止
        rbm.saveRBMOutToFile("rbmOut.txt"); //将RBM的输出层以及标签保存到文件
        uint rbmOutSize = hideUnits[sizeof(hideUnits) / sizeof(hideUnits[0]) - 1];
        cout << endl;
        //将RBM抽取的特征送到ANN中进行分类
        ANN ann(rbmOutSize, tt.n_out, 1); //输入数据大小,输出层神经元个数,隐层层数
        ann.loadTrainSet("rbmOut.txt"); //读取RBM的输出作为ANN的训练数据
        ann.train(0.001, 1000000);
        getImgPredictOut(rbm, ann);
    } catch (const logic_error& err) {
        cout << "\r---error:" << err.what() << endl;
    } catch (...) {
        cout << "\nOops, there are some jokes in the runtime,"
             "I am lost in the jungle\\(╯-╰)/" << endl;
    }
    cin.get();
    return 0;
}
