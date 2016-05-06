#include "RBM.h"
#include "Math_Util.hpp"
#include <iostream>
using namespace std;

int main()
{
    cout << Math_Util::getDateTime() << endl;
    SetText(FG_HL | FG_G | FG_B);
    try {
        int hideUnits[] = { 100, 25 };
        RBM rbm(784, hideUnits);

        cout << "\r>>>正在载入训练数据和测试数据...";
        time_t t_start = clock();
        rbm.loadTrain("./train_test/train0123.txt", 400);
        cout << "\r载入训练和测试数据耗时: " << (clock() - t_start) << "ms" << endl;

        t_start = clock();
        rbm.train(0.02, 10000);  //允许误差和最大代数，任意一个满足则停止
        cout << "\n演化共耗时: " << (clock() - t_start) / CLK_TCK << " s" << endl;
    } catch (const logic_error &err) {
        cout << "\r---error:" << err.what() << endl;
    } catch (...) {
        cout << "\nOops, there are some jokes in the runtime \\(╯-╰)/" << endl;
    }
    cin.get();
    return 0;
}
