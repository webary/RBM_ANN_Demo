#ifndef _RBM_H_
#define _RBM_H_

#include <string>
#include <vector>

using std::vector;
using std::string;

typedef unsigned uint;
typedef vector<float> vectorF;
typedef vector<vectorF> vectorF2D;
typedef vector<vectorF2D> vectorF3D;
typedef vector<unsigned> vectorU;

/**
    受限玻尔兹曼机类,包含一层或多层玻尔兹曼机的属性和操作
**/
class RBM {
public:
    typedef struct {
        vectorF data;        //玻尔兹曼机的输入数据部分
        int tag;             //玻尔兹曼机的输入标签(真or假 or ...)
    } RBMInput;              //玻尔兹曼机输入类型
    typedef struct {
        vectorF3D weight;    //权重
        vectorF2D hbias;     //偏置参数，隐层中每个神经元有一个偏置[正向偏置]
        vectorF2D vbias;     //偏置参数，隐层中每个神经元有一个偏置[反向偏置]
        float fitValue;      //适应值,该个体训练的实际误差,即每一维差值之和
    } RBMIndividual;         //定义个体类型,包含所有可训练参数的集合

    //参数分别是[输入数据维数, 如果是图像是指平铺后大小],
    //[内部网络每层的神经元个数], [参与演化个体数]
    RBM(uint _inputSize, const vectorU& hiddenSizes, double learnRate = .5, uint _popSize = 1)
    {
        init(_inputSize, hiddenSizes, learnRate, _popSize);
    }

    template<int n>
    RBM(uint _inputSize, const int(&hiddenSizes)[n], double learnRate = .5, uint _popSize = 1)
    {
        init(_inputSize, vector<uint>(hiddenSizes, hiddenSizes + n), learnRate, _popSize);
    }
    virtual ~RBM() {}
    //从文件file载入size组训练集数据,divideToTest标记是否将部分数据作为测试集
    void loadTrainSet(const string& file, uint size = 0, bool divideToTest = 1);
    //从文件载入测试数据
    void loadTestSet(const string& file, uint size = 0, bool haveTag = 0);
    //开始训练
    void train(double permitError = 0.05, uint maxGens = 100000);
    //将RBM最后一层的数据输出到文件
    void saveRBMOutToFile(const string& file);
    //得到RBM的最后一层的输出
    vector<RBMInput> getRBMOut();
    //从训练集中随机抽取ratio比例数据作为测试集[训练集中这些数据将剔除]
    void randomDivideTrainToTest(double ratio);
protected:
    //初始化网络结构和参数
    void init(uint _inputSize, const vectorU& hiddenSizes, double learnRate = .5, uint _popSize = 1);
    //前向传递
    void forward(const vectorF& vis, vectorF& hide, const vectorF2D& wt, const vectorF& b);
    //反向传播
    void backward(vectorF& vis, const vectorF& hide, const vectorF2D& wt, const vectorF& b);
    //调整权值和偏置
    double adjust_hvh(RBMIndividual& _rbmPop, uint hideIndex);
    //获取个体_rbmPop的在dataSet数据上第hideindex层的适应值
    float getFitValue(const RBMIndividual& _rbmPop, const vector<RBMInput>& dataSet, uint hideIndex);
    //从预存文件中读入演化参数
    bool loadParam(const string& file);
    //将各层的特征图打印到文件中
    void printParamToFile(const string& file, bool onlyBest = 0);
    //将最好的反演结果输出到文件
    void saveBestReTrain(const string& file);
    //查找当前最好个体
    uint findBestPop();
protected:
    uint inputSize;          //输入数据的大小,即维数
    //vectorF visible;       //visible层,即显层,维数即输入数据的大小[note:由训练集取代]
    vectorF2D hidden;        //hidden层,即隐层(可能有多层,所以又加一维)
    vector<RBMInput> trainSet;   //训练集-包含若干输入数据和标签信息
    vector<RBMInput> testSet;    //测试集-包含若干输入数据(也可包含标签信息)
    vector<RBMIndividual> rbmPop;//个体集合
    float learningRate;          //学习率
    uint bestPopIndex;           //最好个体的索引
};

#ifndef _MSC_VER //兼容非项目编译环境
#include "RBM.cpp"
#endif

#endif //_RBM_H_
