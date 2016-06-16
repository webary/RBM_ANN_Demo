#include "RBM.h"
#include "Math_Util.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdexcept> //logic_error
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
        if (MessageBox(0, "there is a prefetch evolution file, load it?", "prefetch",
                       MB_YESNO | MB_ICONQUESTION) == IDYES)
            loadParam("param.txt");
    }
    bestPopIndex = 0;
}
//从文件file载入size组训练集数据,divideToTest标记是否将部分数据作为测试集
void RBM::loadTrainSet(const string& file, uint size, bool divideToTest)
{
    if (file == "")
        return;
    ifstream loadFile(file.c_str());
    if (loadFile.is_open()) {
        cout << "\r\t\t\t\t\t\t\r>>loading train set...";
        clock_t t1 = clock();
        trainSet.clear();
        if (size > 0)
            trainSet.reserve(size);
        uint i, j;
        RBMInput input = { vectorF(inputSize), 0 };
        for (i = 0; i < size || size == 0; ++i) {
            for (j = 0; j < input.data.size(); ++j)
                if (!(loadFile >> input.data[j]))
                    break;
            //该组数据没有读取完整,则不加入训练集
            if (j < input.data.size() || !(loadFile >> input.tag))
                break;
            trainSet.push_back(input);
        }
        loadFile.close();
        double elapsed_ms = 1.0 * (clock() - t1) / CLOCKS_PER_SEC;
        if (divideToTest) { //随机将部分数据移至测试集
            testSet.resize(trainSet.size() / 5);
            for (i = 0; i < testSet.size(); ++i) {
                j = rand() % trainSet.size();
                testSet[i] = trainSet[j];
                trainSet.erase(trainSet.begin() + j);
            }
            cout << "\r>>succeed to load " << trainSet.size() << " train sets, "
                 << "and generate " << testSet.size() << " test sets!("
                 << elapsed_ms << " s)" << endl;
        } else {
            cout << "\r>>succeed to load " << trainSet.size() << " train sets!("
                 << elapsed_ms << " s)" << endl;
        }
    } else {
        string msg = "Failed to load data set file '" + file + "', please check"
                     " if it exists or has access to read!\n";
        throw logic_error(msg);
    }
}
//从文件载入测试数据
void RBM::loadTestSet(const string& file, uint size, bool haveTag)
{
    if (file == "")
        return;
    ifstream loadFile(file.c_str());
    if (loadFile.is_open()) {
        cout << "\r\t\t\t\t\t\t\r>>loading test set...";
        testSet.clear();
        if (size > 0)
            testSet.reserve(size);
        uint i, j;
        RBMInput input = { vectorF(inputSize), 0 };
        for (i = 0; i < size || size == 0; ++i) {
            for (j = 0; j < input.data.size(); ++j)
                if (!(loadFile >> input.data[j]))
                    break;
            //该组数据没有读取完整，则删掉
            if (j < input.data.size() || haveTag && !(loadFile >> input.tag))
                break;
            testSet.push_back(input);
        }
        loadFile.close();
        cout << "\r>>succeed to load " << testSet.size() << " test sets!" << endl;
    } else {
        string msg = "Failed to load data set file '" + file + "', please check"
                     " if it exists or has access to read!\n";
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
        cout << "\r>>loading prefetch evolution file...";
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

        //对比分歧算法(contrastive divergence,CD),也叫对比散度
        //修正该层所有权值
        for (i = 0; i < new_vis.size(); ++i)
            for (j = 0; j < new_hide.size(); ++j)
                _rbmPop.weight[h][i][j] += learningRate / trainSet.size() *
                                           (hidden[h][j] * lastVis[i] - new_hide[j] * new_vis[i]);
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
    if (dataSet.size() == 0) { //还没有数据
        cout << "there are some errors in data set, please check it!\n in "
             << __FILE__ << ":" << __LINE__ << endl;
        return 1;
    }
    vectorF new_vis;  //反演得到的显层 h->v'
    float fitValue = 0;
    if (hideIndex == 0)
        new_vis = dataSet[0].data;
    else
        new_vis = hidden[hideIndex - 1];
    for (uint ts = 0, h; ts < dataSet.size(); ++ts) {
        forward(dataSet[ts].data, hidden[0], _rbmPop.weight[0], _rbmPop.hbias[0]);
        for (h = 1; h <= hideIndex; ++h) //更新隐层h
            forward(hidden[h - 1], hidden[h], _rbmPop.weight[h], _rbmPop.hbias[h]);
        h = hideIndex;
        backward(new_vis, hidden[h], _rbmPop.weight[h], _rbmPop.vbias[h]);
        //对比分歧算法(contrastive divergence,CD),也叫对比散度
        for (uint j = 0; j < new_vis.size(); ++j) {
            float err = 0;
            if (h == 0)
                err = Math_Util::myAbs(dataSet[ts].data[j] - new_vis[j]);
            else
                err = Math_Util::myAbs(hidden[h - 1][j] - new_vis[j]);
            fitValue += err;
        }
    }
    return fitValue / (dataSet.size() * new_vis.size());
}
//开始训练
void RBM::train(double permitError, uint maxGens)
{
    setGreen();
    if (trainSet.size() <= 0) {
        cout << "please load train set first!" << endl;
        return;
    }
    uint i, gen;
    createFolder("RBM_Gen_TrainRight");   //创建文件夹
    char tmpbuf[128];
    SPRINTF(tmpbuf, "RBM_Gen_TrainRight/%s.txt", Math_Util::getDateTime(0, '.').c_str());
    ofstream saveRight(tmpbuf);
    clock_t train_start = clock();
    for (uint h = 0; h < hidden.size(); ++h) {
        saveRight << ">>>hide<" << h << ">\nGen\ttrain accuracy\ttest accuracy" << endl;
        setBlue();
        cout << "\n>>>hide<" << h << ">\nGen\ttrain accuracy\ttest accuracy\t"
             "average elapsed(ms)" << endl;
        setGreen();
        float lastError = 1, testRight = 0, trainError;
        clock_t t_start = clock(), t_end = 0, lastGen = -1;
        for (gen = 0; gen <= maxGens; ++gen) {
            for (i = 0; i < rbmPop.size(); ++i)
                adjust_hvh(rbmPop[i], h);
            uint best = findBestPop();
            trainError = rbmPop[best].fitValue; //当前最好个体在训练集上的误差
            t_end = clock();
            if (t_end - t_start > 200 || gen % 5 == 0) {  //不断刷新当前进度
                testRight = 1 - getFitValue(rbmPop[best], testSet, h);
                double elapsed_ms = 1000.0 * (t_end - t_start) / CLOCKS_PER_SEC;
                cout << "\r" << setw(40) << " " << "\r" << gen << "\t"
                     << setw(16) << setiosflags(ios::left) << 1 - trainError
                     << setw(16) << testRight << elapsed_ms / (gen - lastGen);
                if ((gen % 100 == 0 && lastError - trainError > 0.005)
                        || lastError - trainError > 0.02) {
                    cout << "\nsaving the parameters of the best pop...";
                    saveRight << gen << "\t" << setw(16) << setiosflags(ios::left)
                              << 1 - trainError << setw(24) << testRight << endl;
                    lastError = trainError;
                    cout << "\r\t\t\t\t\t\t\r";
                }
                lastGen = gen;
                t_start = t_end;
            }
            if (trainError < permitError) {
                saveRight << gen << "\t" << setw(16) << setiosflags(ios::left)
                          << 1 - trainError << setw(24) << testRight << endl;
                break;
            }
            if (checkKeyDown() == 27 &&
                    MessageBox(0, "Do you want to stop the evolution?",
                               "stop", MB_YESNO | MB_ICONQUESTION) == IDYES)
                break;
        } //for (gen)
    } //for (h)
    double elapsed_s = 1.0 * (clock() - train_start) / CLOCKS_PER_SEC;
    cout << "\ntime elapsed of evolving: " << elapsed_s << " s" << endl;
    saveRight.close();
    printParamToFile("param.txt", 1);
    saveBestReTrain("reTrain.txt");
}
//将RBM最后一层的数据输出到文件
void RBM::saveRBMOutToFile(const string& file,bool enableTrainSet, bool enableTestset)
{
    ofstream fileOut(file.c_str());
    if (fileOut.is_open()) {
        vector<RBMInput> rbmOut = getRBMOut(enableTrainSet, enableTestset);
        for (uint s = 0; s < rbmOut.size(); ++s) {
            for (uint i = 0; i < rbmOut[s].data.size(); ++i)
                fileOut << setw(8) << setiosflags(ios::left) << setprecision(3)
                        << setiosflags(ios::fixed) << rbmOut[s].data[i];
            fileOut << '\t' << rbmOut[s].tag << endl;
        }
        fileOut.close();
    }
}
//得到RBM的最后一层的输出
vector<RBM::RBMInput> RBM::getRBMOut(bool enableTrainSet, bool enableTestset)
{
    size_t outSize = 0;
    if (enableTrainSet)
        outSize += trainSet.size();
    if (enableTestset)
        outSize +=  testSet.size();
    vector<RBMInput> rbmOut(outSize);
    uint best = findBestPop(), i, h;
    if (enableTrainSet) {
        for (i = 0; i < trainSet.size(); ++i) {
            forward(trainSet[i].data, hidden[0], rbmPop[best].weight[0], rbmPop[best].hbias[0]);
            for (h = 1; h < hidden.size(); ++h) //更新隐层h
                forward(hidden[h - 1], hidden[h], rbmPop[best].weight[h], rbmPop[best].hbias[h]);
            rbmOut[i] = { hidden[h - 1], trainSet[i].tag };
        }
    }
    if (enableTestset) {
        uint outStart = enableTrainSet ? trainSet.size() : 0;
        for (i = 0; i < testSet.size(); ++i) {
            forward(testSet[i].data, hidden[0], rbmPop[best].weight[0], rbmPop[best].hbias[0]);
            for (h = 1; h < hidden.size(); ++h) //更新隐层h
                forward(hidden[h - 1], hidden[h], rbmPop[best].weight[h], rbmPop[best].hbias[h]);
            rbmOut[i + outStart] = { hidden[h - 1], testSet[i].tag };
        }
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
    cout << "\rsaving best rebuild result...";
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
    ShellExecute(0, "open", "RE2JPG.exe", "0", 0, SW_HIDE);
    //system("RE2JPG.exe 0");
    cout << "\r\t\t\t\t\t\t\r";
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

/** //usage:
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
    SetText(FG_HL | FG_G | FG_B);
    try {
        int hideUnits[] = { 100, 25 };
        RBM rbm(784, hideUnits);
        rbm.loadTrainSet(tt.file, tt.n_train);
        rbm.train(0.02, 10000);  //允许误差和最大代数,任意一个满足则停止
    } catch (const logic_error& err) {
        cout << "\r---error:" << err.what() << endl;
    } catch (...) {
        cout << "\nOops, there are some jokes in the runtime \\(╯-╰)/" << endl;
    }
    cin.get();
    return 0;
}
*/
