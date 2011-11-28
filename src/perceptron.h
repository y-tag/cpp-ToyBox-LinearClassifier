#ifndef TOYBOX_PERCEPTRON_H
#define TOYBOX_PERCEPTRON_H

#include <vector>

#define WVector FVector
//#define WVector SVector

namespace toybox {

class SVector;
class FVector;

enum ModelType {
  BIN = 0, MC,
  MODEL_NUM
};

struct Parameter {
  ModelType model_type;
  int T;
  bool verbose;
};

class Perceptron {
  public:
    Perceptron();
    ~Perceptron();
    void Train(const std::vector<SVector> &x_vec,
               const std::vector<int> &y_vec,
               const struct Parameter &param);
    int Predict(const SVector &x, double *predict) const;
    int Save(const char *filename) const;
    int Load(const char *filename);
    int GetLabels(int *labels) const;
    int lnum() const { return lnum_; };
    int fnum() const { return fnum_; };
  private:
    int lnum_;
    int fnum_;
    int *labels_;
    std::vector<WVector*> weight_;
    Perceptron(const Perceptron&);
    void operator=(const Perceptron&);
};

} // namespace toybox

#endif // TOYBOX_PERCEPTRON_H
