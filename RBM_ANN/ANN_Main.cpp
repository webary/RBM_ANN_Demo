#include "ANN.h"
#include "Math_Util.hpp"
#include <iostream>
using namespace std;

typedef struct {
    string file; //训练集文件
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

int main()
{
    TrainTxt& tt = tt__[1]; ///通过修改序号载入不同的训练集
    cout << Math_Util::getDateTime() << "\t" << tt.file << "\t" << tt.n_train << endl;
    try {
        ANN ann(784, tt.n_out, 1); //输入数据维数, 输出层结点个数, 隐藏层层数
        ann.loadTrainSet(tt.file, tt.n_train);
        ann.train(0.01, 1000000);
    } catch (const logic_error& err) {
        cout << "\r---error:" << err.what() << endl;
    } catch (...) {
        cout << "Oops, I am so sorry to print this!" << endl;
    }
    cin.get();
    return 0;
}
