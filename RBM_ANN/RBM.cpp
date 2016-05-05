#include "RBM.h"
#include "Win_Util.h"
#include "Math_Util.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdexcept> //logic_error
#include <conio.h>   //getch(), kbhit()
#include <direct.h>  //_mkdir
using namespace std;

//初始化网络结构和参数
void RBM::init(uint _inputSize, const vectorU& hiddenSizes, double learnRate, uint _popSize)
{
    Math_Util::setSrand(); //设置随机种子
    inputSize = _inputSize;
    learningRate = (float)learnRate;
    hidden.resize(hiddenSizes.size());//设置隐层层数
    uint i, j, k, index;
    for (i = 0; i < hidden.size(); ++i)
        hidden[i].resize(hiddenSizes[i]);
    //以下部分初始化RBM的个体
    rbmPop.resize(_popSize);
    for (index = 0; index < rbmPop.size(); ++index) {
        uint v_size = inputSize, h_size = 0;
        //初始化权重. 范围是经验值,可以在后期调整
        rbmPop[index].weight.resize(hiddenSizes.size());
        for (i = 0; i < rbmPop[index].weight.size(); ++i) {
            h_size = hidden[i].size();
            rbmPop[index].weight[i].resize(v_size);
            for (j = 0; j < v_size; ++j) {
                rbmPop[index].weight[i][j].resize(h_size);
                for (k = 0; k < h_size; ++k)
                    rbmPop[index].weight[i][j][k] = Math_Util::randFloat(-1, 1);
            }
            v_size = h_size;    //隐层将变为显层
        }
        //初始化偏置参数,每个隐层对应一个偏置,初始化为0. 范围是经验值
        rbmPop[index].hbias.resize(hiddenSizes.size());
        rbmPop[index].vbias.resize(hiddenSizes.size());
        rbmPop[index].vbias[0].resize(inputSize, 0);
        for (i = 0; i < hiddenSizes.size(); ++i) {
            rbmPop[index].hbias[i].resize(hiddenSizes[i], 0);
            if (i < hiddenSizes.size() - 1)
                rbmPop[index].vbias[i + 1].resize(hiddenSizes[i], 0);
        }
    }
    //从预存参数文件中读入演化参数
    ifstream testExist("param.txt");
    if (testExist.is_open()) {
        testExist.close();
        if (MessageBox(0, "检测到已存在演化预存文件，是否载入？", "温馨提示", MB_YESNO) == IDYES)
            loadParam("param.txt");
    }
    bestPopIndex = 0;
}
//从文件file载入size组训练集数据,divideToTest标记是否将部分数据作为测试集
void RBM::loadTrain(const string& file, uint size, bool divideToTest)
{
    if (file == "")
        return;
    ifstream loadFile(file.c_str());
    if (loadFile.is_open()) {
        trainSet.clear();
        if (size>0)
            trainSet.reserve(size);
        uint i, j;
        clock_t t1 = clock();
        RBMInput input = { vectorF(inputSize), 0 };
        for (i = 0; i < size || size == 0; ++i) {
            for (j = 0; j < input.data.size(); ++j)
                if (!(loadFile >> input.data[j]))
                    break;
            //该组数据没有读取完整,则不加入训练集
            if (!(loadFile >> input.tag))
                break;
            trainSet.push_back(input);
        }
        loadFile.close();
        if (divideToTest) { //随机将部分数据移至测试集
            testSet.resize(trainSet.size() / 5);
            for (i = 0; i < testSet.size(); ++i) {
                j = rand() % trainSet.size();
                testSet[i] = trainSet[j];
                trainSet.erase(trainSet.begin() + j);
            }
            cout << "\r>>成功载入" << trainSet.size() << "组训练集, 并生成"
                << testSet.size() << "组测试集(" << (clock() - t1) / 1000.0 << "s)\n";
        } else {
            cout << "\r>>成功载入" << trainSet.size() << "组训练集(" << (clock() - t1) / 1000.0 << "s)\n";
        }
    } else {
        string msg = "载入数据集文件'" + file + "失败，请检查该文件是否存在，或有权限读取\n";
        throw logic_error(msg);
    }
}
//从文件载入测试数据
void RBM::loadTest(const string& file, uint size, bool haveTag)
{
    if (file == "")
        return;
    ifstream loadFile(file.c_str());
    if (loadFile.is_open()) {
        testSet.clear();
        if (size>0)
            testSet.reserve(size);
        uint i, j;
        RBMInput input = { vectorF(inputSize), 0 };
        for (i = 0; i < size || size == 0; ++i) {
            for (j = 0; j < input.data.size(); ++j)
                if (!(loadFile >> input.data[j]))
                    break;
            //该组数据没有读取完整，则删掉
            if (haveTag && !(loadFile >> input.tag))
                break;
            testSet.push_back(input);
        }
        loadFile.close();
        cout << "\r\t\t\t\t\t\r>>成功载入" << testSet.size() << "组测试集" << endl;
    } else {
        string msg = "载入数据集文件'" + file + "失败，请检查该文件是否存在，或有权限读取\n";
        throw logic_error(msg);
    }
}
//从预存文件中读入演化参数
bool RBM::loadParam(const string& file)
{
    if (file == "")
        return false;
    ifstream loadFile(file.c_str());
    if (loadFile.is_open()) {
        cout << ">>>正在读取预存参数文件...";
        uint i, j, k;
        try {
            for (i = 0; i < rbmPop[0].weight.size(); ++i) {
                for (j = 0; j < rbmPop[0].weight[i].size(); ++j) {
                    for (k = 0; k < rbmPop[0].weight[i][j].size(); ++k)
                        loadFile >> rbmPop[0].weight[i][j][k];
                }
            }
            for (i = 0; i < rbmPop[0].hbias.size(); ++i)
                for (j = 0; j < rbmPop[0].hbias[i].size(); ++j)
                    loadFile >> rbmPop[0].hbias[i][j];
            for (i = 0; i < rbmPop[0].vbias.size(); ++i)
                for (j = 0; j < rbmPop[0].vbias[i].size(); ++j)
                    loadFile >> rbmPop[0].vbias[i][j];
            loadFile.close();
        } catch (...) {
            return false;
        }
        return true;
    }
    return false;
}
//正向传递 v * wt + b => h
void RBM::forward(const vectorF& vis, vectorF& hide, const vectorF2D& wt, const vectorF& b)
{
    uint i, j;
    for (j = 0; j < hide.size(); ++j) {
        hide[j] = 0;
        for (i = 0; i < vis.size(); ++i)
            hide[j] += vis[i] * wt[i][j];
        hide[j] = (float)Math_Util::sigmoid(hide[j] + b[j]);
    }
}
//反向传播 h * wt' + b => v
void RBM::backward(vectorF& vis, const vectorF& hide, const vectorF2D& wt, const vectorF& b)
{
    uint i, j;
    for (i = 0; i < vis.size(); ++i) {
        vis[i] = 0;
        for (j = 0; j < hide.size(); ++j)
            vis[i] += hide[j] * wt[i][j];
        vis[i] = (float)Math_Util::sigmoid(vis[i] + b[i]);
    }
}
//调整权值和偏置
double RBM::adjust_hvh(RBMIndividual& _rbmPop, uint hideIndex)
{
    if (hideIndex >= hidden.size()) //输入的序号有误
        return 1;
    uint ts, i, j, h;
    vectorF new_vis;  //反演得到的显层 h->v'
    vectorF new_hide; //重新得到的隐层 v'->h'
    if (hideIndex == 0)
        new_vis = trainSet[0].data;
    else
        new_vis = hidden[hideIndex - 1];
    for (ts = 0; ts < trainSet.size(); ++ts) {
        forward(trainSet[ts].data, hidden[0], _rbmPop.weight[0], _rbmPop.hbias[0]);
        for (h = 1; h <= hideIndex; ++h) //更新隐层h
            forward(hidden[h - 1], hidden[h], _rbmPop.weight[h], _rbmPop.hbias[h]);
        h = hideIndex;
        vectorF lastVis = (h == 0) ? trainSet[ts].data : hidden[h - 1]; //原始显层
        new_hide = hidden[h]; //保存重新计算得到的隐层
        backward(new_vis, hidden[h], _rbmPop.weight[h], _rbmPop.vbias[h]); //h->v'
        forward(new_vis, new_hide, _rbmPop.weight[h], _rbmPop.hbias[h]); //v'->h'

        //修正该层所有权值
        for (i = 0; i < new_vis.size(); ++i)
            for (j = 0; j < new_hide.size(); ++j)
                _rbmPop.weight[h][i][j] += learningRate \
                * (hidden[h][j] * lastVis[i] - new_hide[j] * new_vis[i]) / trainSet.size();
        //修正正向偏置
        for (j = 0; j < new_hide.size(); ++j)
            _rbmPop.hbias[h][j] += learningRate * (hidden[h][j] - new_hide[j]) / trainSet.size();
        //修正反向偏置
        for (i = 0; i < new_vis.size(); ++i)
            _rbmPop.vbias[h][i] += learningRate * (lastVis[i] - new_vis[i]) / trainSet.size();
    }
    _rbmPop.fitValue = getFitValue(_rbmPop, trainSet, hideIndex);
    return _rbmPop.fitValue;
}
//获取个体_rbmPop在dataSet数据上第hideindex层的适应值
float RBM::getFitValue(const RBMIndividual& _rbmPop, const vector<RBMInput>& dataSet, uint hideIndex)
{
    uint ts, j, h;
    vectorF new_vis;  //反演得到的显层 h->v'
    float fitValue = 0;
    if (hideIndex == 0)
        new_vis = dataSet[0].data;
    else
        new_vis = hidden[hideIndex - 1];
    for (ts = 0; ts < dataSet.size(); ++ts) {
        forward(dataSet[ts].data, hidden[0], _rbmPop.weight[0], _rbmPop.hbias[0]);
        for (h = 1; h <= hideIndex; ++h) //更新隐层h
            forward(hidden[h - 1], hidden[h], _rbmPop.weight[h], _rbmPop.hbias[h]);
        h = hideIndex;
        backward(new_vis, hidden[h], _rbmPop.weight[h], _rbmPop.vbias[h]);
        //对比分歧
        for (j = 0; j < new_vis.size(); ++j) {
            float err = 0;
            if (h == 0)
                err = Math_Util::myAbs(dataSet[ts].data[j] - new_vis[j]);
            else
                err = Math_Util::myAbs(hidden[h - 1][j] - new_vis[j]);
            fitValue += err;
        }
    }
    if (ts == 0) {     //还没有训练数据
        cout << "训练数据有误，请检查!"<<__FILE__<<":" << __LINE__ << endl;
        return 1;
    }
    return fitValue / (ts * new_vis.size());
}
//开始训练
void RBM::train(double permitError, uint maxGens)
{
    setGreen();
    if (trainSet.size() <= 0) {
        cout << "请先载入训练集再开始训练！" << endl;
        throw logic_error("请先载入训练集再开始训练");
    }
    uint i, gen;
    _mkdir("RBM_Gen_TrainRight");   //创建文件夹
    char tmpbuf[128];
    SPRINTF(tmpbuf, "RBM_Gen_TrainRight/%s.txt", Math_Util::getDateTime(0, '.').c_str());
    ofstream saveRight(tmpbuf);
    for (uint h = 0; h < hidden.size(); ++h) {
        saveRight << ">>>隐层<" << h << ">\n代数\t训练正确率\t测试正确率" << endl;
        setBlue();
        cout << endl << ">>>隐层<" << h << ">\n代数\t训练正确率\t测试正确率\t每代平均耗时(ms)" << endl;
        setGreen();
        float lastError = 1, testRight = 0;
        unsigned t_start = clock(), t_end = 0, lastGen = -1;
        for (gen = 0; gen <= maxGens; ++gen) {
            for (i = 0; i < rbmPop.size(); ++i)
                adjust_hvh(rbmPop[i], h);
            uint best = findBestPop();
            t_end = clock();
            if (t_end - t_start > 200 || gen % 5 == 0) {  //不断刷新当前进度
                testRight = 1 - getFitValue(rbmPop[best], testSet, h);
                cout << "\r" << setw(40) << " " << "\r" << gen <<"\t"<< setw(16)
                    << setiosflags(ios::left) << 1 - rbmPop[best].fitValue
                    << setw(16) << testRight << 1.*(t_end - t_start) / (gen - lastGen);
                if ((gen % 100 == 0 && lastError - rbmPop[best].fitValue > 0.005)
                    || lastError - rbmPop[best].fitValue > 0.04) {
                    cout << endl << "正在保存当前最好参数...";
                    saveRight << gen << "\t" << setw(16) << setiosflags(ios::left)
                        << 1 - rbmPop[best].fitValue << setw(24) << testRight << endl;
                    lastError = rbmPop[best].fitValue;
                    cout << "\r\t\t\t\t\r";
                }
                lastGen = gen;
                t_start = t_end;
            }
            if (rbmPop[best].fitValue < permitError) {
                saveRight << gen << "\t" << setw(16) << setiosflags(ios::left)
                    << 1 - rbmPop[best].fitValue << setw(24) << testRight << endl;
                break;
            }
            if (_kbhit() && 27 == _getch() && MessageBox(0, "你按了ESC键,是否要结束演化过程?", "温馨提示", MB_YESNO) == IDYES)
                break;
        } //for (gen)
    } //for (h)
    saveRight.close();
    cout << endl;
    printParamToFile("param.txt", 1);
    saveBestReTrain("reTrain.txt");
}
//将RBM最后一层的数据输出到文件
void RBM::saveRBMOutToFile(const std::string& file)
{
    ofstream fileOut(file.c_str());
    if (fileOut.is_open()) {
        vector<RBMInput> rbmOut = getRBMOut();
        for (uint s = 0; s < rbmOut.size(); ++s) {
            for (uint i = 0; i < rbmOut[s].data.size(); ++i)
                fileOut << setw(8) << setiosflags(ios::left) << setprecision(3) << setiosflags(ios::fixed) << rbmOut[s].data[i];
            fileOut << "\t" << rbmOut[s].tag << endl;
        }
        fileOut.close();
    }
}
//得到RBM的最后一层的输出
vector<RBM::RBMInput> RBM::getRBMOut()
{
    vector<RBMInput> rbmOut(trainSet.size()+testSet.size());
    uint best = findBestPop(), i, h;
    for (i = 0; i < trainSet.size(); ++i) {
        forward(trainSet[i].data, hidden[0], rbmPop[best].weight[0], rbmPop[best].hbias[0]);
        for (h = 1; h < hidden.size(); ++h) //更新隐层h
            forward(hidden[h - 1], hidden[h], rbmPop[best].weight[h], rbmPop[best].hbias[h]);
        rbmOut[i] = { hidden[h - 1],trainSet[i].tag };
    }
    for (i = 0; i < testSet.size(); ++i) {
        forward(testSet[i].data, hidden[0], rbmPop[best].weight[0], rbmPop[best].hbias[0]);
        for (h = 1; h < hidden.size(); ++h) //更新隐层h
            forward(hidden[h - 1], hidden[h], rbmPop[best].weight[h], rbmPop[best].hbias[h]);
        rbmOut[i+trainSet.size()] = { hidden[h - 1],testSet[i].tag };
    }
    return rbmOut;
}
//从训练集中随机抽取ratio比例数据作为测试集[训练集中这些数据将剔除]
void RBM::randomDivideTrainToTest(double ratio)
{
    uint testSize = (uint)(trainSet.size() * ratio); //测试集大小
    if (testSize < 1 || testSize > trainSet.size()) //如果参数范围不正确，默认划分1/4
        testSize = (int)(trainSet.size() * 0.2);
    testSet.resize(testSize);
    for (uint i = 0; i < testSize; ++i) {    //随机将部分数据移至测试集
        uint j = rand() % trainSet.size();
        testSet[i] = trainSet[j];
        trainSet.erase(trainSet.begin() + j);
    }
}
//将演化参数及中间值输出到文件中
void RBM::printParamToFile(const string& file, bool onlyBest)
{
    ofstream outFile(file.c_str());
    if (outFile.is_open()) {
        uint i, j, k, index;
        for (index = 0; index < rbmPop.size(); ++index) {
            if (onlyBest)
                index = bestPopIndex;
            for (i = 0; i < rbmPop[index].weight.size(); ++i) {
                for (j = 0; j < rbmPop[index].weight[i].size(); ++j) {
                    for (k = 0; k < rbmPop[index].weight[i][j].size(); ++k)
                        outFile << setw(10) << rbmPop[index].weight[i][j][k] << " ";
                    outFile << endl;
                }
                outFile << endl;
            }
            for (i = 0; i < rbmPop[index].hbias.size(); ++i)
                for (j = 0; j < rbmPop[index].hbias[i].size(); ++j)
                    outFile << rbmPop[index].hbias[i][j] << "  ";
            outFile << endl;
            for (i = 0; i < rbmPop[index].vbias.size(); ++i)
                for (j = 0; j < rbmPop[index].vbias[i].size(); ++j)
                    outFile << rbmPop[index].vbias[i][j] << "  ";
            outFile << endl;
            if (onlyBest)
                break;
        }
        outFile << "best fitValue:" << rbmPop[bestPopIndex].fitValue << endl;
        outFile.close();
    }
}
//保存最好的反演结果
void RBM::saveBestReTrain(const string& file)
{
    cout << "\r正在保存最好反演结果...";
    uint best = findBestPop(), ts, h;
    vector<RBMInput> reTrain = trainSet;
    //最好个体反演得到重构的输入图
    for (ts = 0; ts < trainSet.size(); ++ts) {
        //正向计算
        forward(reTrain[ts].data, hidden[0], rbmPop[best].weight[0], rbmPop[best].hbias[0]);
        for (h = 1; h < hidden.size(); ++h) //更新隐层h
            forward(hidden[h - 1], hidden[h], rbmPop[best].weight[h], rbmPop[best].hbias[h]);
        //反向计算
        for (--h; h > 0; --h)
            backward(hidden[h - 1], hidden[h], rbmPop[best].weight[h], rbmPop[best].vbias[h]);
        backward(reTrain[ts].data, hidden[0], rbmPop[best].weight[0], rbmPop[best].vbias[0]);
    }
    //保存重构输入图数据
    ofstream outFile(file.c_str());
    if (outFile.is_open()) {
        uint i, j;
        for (i = 0; i < reTrain.size(); ++i) {
            for (j = 0; j < reTrain[i].data.size(); ++j)
                outFile << setw(9) << reTrain[i].data[j] << " ";
            outFile << endl;
        }
        outFile.close();
    }
    //将数据转换为图像保存
    ShellExecute(0, "open", "RE2JPG.exe","0", 0, SW_HIDE);
    //system("RE2JPG.exe 0");
    cout << "\r\t\t\t\r";
}
//查找当前最好个体
uint RBM::findBestPop()
{
    bestPopIndex = 0;
    for (uint i = 1; i < rbmPop.size(); ++i)
        if (rbmPop[i].fitValue < rbmPop[bestPopIndex].fitValue)
            bestPopIndex = i;
    return bestPopIndex;
}

/* usage:
int main() {
    cout << RBM::getDateTime().c_str() << endl;
    SetText(FG_HL | FG_G | FG_B);
    try {
        //Param::loadINI("set.ini","RBM");
        int hideUnits[] = { 100, 25 };
        RBM rbm(784, hideUnits);

        cout << "\r>>>正在载入训练数据和测试数据...";
        time_t t_start = clock();
        rbm.loadTrain("D:/train_test/train0123.txt", 100);
        cout << "\r载入训练和测试数据耗时: " << (clock() - t_start) << "ms" << endl;

        t_start = clock();
        rbm.train(0.0001, 10000);    //允许误差和最大代数，任意一个满足则停止
        cout << "\n演化共耗时: " << (clock() - t_start) / 1000.0 << " s" << endl;

        t_start = clock();
        cout << "rbm最后一个隐层的输出为：" << endl;
        vectorF2D rbmOut = rbm.getRBMOut();
        for (uint i = 0; i < rbmOut.size();++i) {
            for (uint j = 0; j < rbmOut[i].size(); ++j)
                cout << rbmOut[i][j] << "  ";
            cout << endl;
        }
        cout << "耗时：" << clock() - t_start << endl;
    }
    catch (const char* str) {
        cout << "\n--error:" << str << endl;
    }
    catch (...) {
        cout << "\nOops, there are some jokes in the runtime \\(╯-╰)/" << endl;
    }
    return 0;
}
*/
