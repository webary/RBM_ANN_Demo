#include "ANN.h"
#include <iostream>
using namespace std;

int main()
{
    try {
        ANN ann(784, 4, 1); //输入数据维数, 输出层结点个数, 隐藏层层数
        ann.loadTrainSet("./train_test/train0123.txt", 500);
        ann.train(0.01, 1000000);
    } catch (const logic_error& err) {
        cout << "\r---error:" << err.what() << endl;
    } catch (...) {
        cout << "Oops, I am so sorry to print this!" << endl;
    }
    cin.get();
    return 0;
}
