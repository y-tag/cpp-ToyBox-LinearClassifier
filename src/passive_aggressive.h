#ifndef TOYBOX_PASSIVEAGGRESSIVE_H
#define TOYBOX_PASSIVEAGGRESSIVE_H

#include <vector>

#define WVector FVector
//#define WVector SVector

namespace toybox {

class SVector;
class FVector;

enum ModelType {
  BIN_PA = 0, BIN_PA1, BIN_PA2,
  MC_PA, MC_PA1, MC_PA2,
  MODEL_NUM
};

struct Parameter {
  ModelType model_type;
  int T;
  double C;
  bool verbose;
};

class PassiveAggressive {
  public:
    PassiveAggressive();
    ~PassiveAggressive();
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
    PassiveAggressive(const PassiveAggressive&);
    void operator=(const PassiveAggressive&);
};

} // namespace toybox

#endif // TOYBOX_PASSIVEAGGRESSIVE_H
