#ifndef TOYBOX_SVMSGD_H
#define TOYBOX_SVMSGD_H

#include <vector>

#define WVector FVector
//#define WVector SVector

namespace toybox {

class SVector;
class FVector;

enum ModelType {
  SGD2_BIN_L1 = 0, SGD2_BIN_L2, SGD2_BIN_LOGLOSS,
  MODEL_NUM
};

struct Parameter {
  ModelType model_type;
  int T;
  int t0;
  double lambda;
  int skip;
  bool verbose;
};

class SVMSGD {
  public:
    SVMSGD();
    ~SVMSGD();
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
    SVMSGD(const SVMSGD&);
    void operator=(const SVMSGD&);
};

} // namespace toybox

#endif // TOYBOX_SVMSGD_H
