#ifndef _ANN_H_
#define _ANN_H_

#include <vector>
#include <string>

using std::vector;
using std::string;

typedef unsigned uint;
typedef vector<float> vectorF;
typedef vector<vectorF> vectorF2D;
typedef vector<vectorF2D> vectorF3D;
typedef vector<unsigned> vectorU;

/**
 * 经典神经网络类
 **/
class ANN
{
public:
    typedef struct {
        vectorF data;  //输入数据部分
        int tag;       //输入标签(真or假 or ...)
    } ANNInput;        //输入数据的结构
    typedef struct {
        vectorF3D weight; //权重参数,隐含层和输出层每个神经元都有一组权值
        vectorF2D bias;   //偏置参数,隐含层和输出层每个神经元都有一个偏置
        float fitValue;   //适应值
    } ANNIndividual;      //神经网络演化个体类型,包含所有可训练参数
    typedef struct {
        vectorF input;    //输入层
        vectorF2D hide;   //隐藏层
        vectorU output;   //输出层
        int tag;          //网络预测得到的标签
    } ANNLayer;           //ANN网络结构

    //@param:输入数据, 输出结点个数, 隐藏层层数
    ANN(const vector<ANNInput>& _inputData, uint outputNums, uint hideLayers = 1);
    //@param:输入数据维数, 输出层结点个数, 隐藏层层数
    ANN(uint inputSize, uint outputNums, uint hideLayers = 1);
    ~ANN() {}
    //从文件file载入size组训练集数据,divideToTest标记是否将部分数据作为测试集
    void loadTrainSet(const string& file, uint size = 0, bool divideToTest = 1);
    //从文件file载入size组测试集数据,haveTag标记是否有标签
    void loadTestSet(const string& file, uint size = 0, bool haveTag = 0);
    //开始训练神经网络 @param: 允许误差 0.0~1.0 ; 最大迭代次数
    void train(double permitError, uint maxGens = 100000);
    //根据训练好的网络,获得测试集预测结果
    vector<int> getTestOut() const;
    //对比测试集的输出,返回正确率---前提是已知每组测试数据的输出
    float compareTestOut() const;
    //从训练集中随机抽取ratio比例数据作为测试集[训练集中这些数据将剔除]
    void randomDivideTrainToTest(double ratio = 0.25);
protected:
    //对网络结构及可演化参数进行初始化
    void init(uint inputSize, uint outputNums, uint hideLayers = 2);
    //计算某组输入数据对应的网络输出 (预测标签)
    uint getANNOut(const ANNInput& _input, const ANNIndividual& indiv) const;
    //计算个体的适应值
    float getFitValue(ANNIndividual& indiv);
    //个体变异并选择更优个体进入下一代
    void mutate(ANNIndividual& _annPop);
    //通过高斯变异方式调整参数---mutate的一种实现方式
    void mutateByGauss(ANNIndividual& tmpPop);
    //返回最优个体的常引用
    const ANNIndividual& getBestPop() const;
protected:
    mutable ANNLayer layer;       //神经结构层
    vector<ANNInput> trainSet;    //神经网络训练数据
    vector<ANNInput> testSet;     //神经网络测试数据
    vector<ANNIndividual> annPop; //多个个体组成的种群

    static const float F_ANN; //神经网络的变异概率
};

#ifndef _MSC_VER //兼容非项目编译环境
#include "ANN.cpp"
#endif  //_MSC_VER

#endif  //_ANN_H_
