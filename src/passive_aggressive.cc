#include "passive_aggressive.h"

#include <cassert>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <fstream>
#include <string>
#include <vector>

#include "vectors.h"

namespace {

int get_lnum_fnum_labels(const std::vector<toybox::SVector> &x_vec,
                         const std::vector<int> &y_vec,
                         int *lnum, int *fnum, std::vector<int> *labels) {
  if (lnum == NULL || lnum == NULL || labels == NULL) {
    return 0;
  }

  int dnum = x_vec.size();
  for (int i = 0; i < dnum; ++i) {
    int j = 0;
    for (j = 0; j < *lnum; ++j) {
      if (y_vec[i] == (*labels)[j]) { break; }
    }
    if (j == *lnum) {
      labels->push_back(y_vec[i]);
      ++(*lnum);
    }
    if (x_vec[i].size() > *fnum) { *fnum = x_vec[i].size(); }
  }

  return 1;
}

template<typename Function>
int train_bin (int T, const std::vector<toybox::SVector> x_vec,
               int *y, bool verbose, Function tau_calc,
               toybox::WVector *weight) {
  if (T <= 0 || y == NULL || weight == NULL) { return 0; }
  int dnum = x_vec.size();

  int t;
  for (t = 0; t < T; ++t) {
    int miss_num = 0;
    for (int i = 0; i < dnum; ++i) {
      double score = weight->Dot(x_vec[i]);
      double loss = 1.0 - y[i] * score;
      if (loss > 0.0) {
        double tau = tau_calc(loss, x_vec[i].snorm());
        weight->Add(x_vec[i], y[i] * tau);
        ++miss_num;
      }
    }
    if (verbose && t % 10 == 0) { fprintf(stderr, "."); }
    if (miss_num == 0) { break; }
  }
  if (verbose) { fprintf(stderr, "done(%d)\n", t); }

  return 1;
}

template<typename Function>
int train_multi (int T, int lnum, int fnum, 
                 const std::vector<toybox::SVector> &x_vec,
                 const std::vector<int> &y_vec,
                 int *labels, bool verbose, Function tau_calc,
                 std::vector<toybox::WVector*> *weight) {
  if (T <= 0 || lnum <= 0 || fnum <= 0 ||
      labels == NULL || weight == NULL) {
    return 0;
  }
  int dnum = x_vec.size();

  int t;
  for (t = 0; t < T; ++t) {
    int miss_num = 0;
    for (int i = 0; i < dnum; ++i) {
      int yl = -1;
      int maxl = -1;
      double max_noty_score = -DBL_MAX;
      double y_score = 0.0;
      for (int l = 0; l < lnum; ++l) {
        double score = (*weight)[l]->Dot(x_vec[i]);
        if (labels[l] == y_vec[i]) {
          yl = l;
          y_score = score;
        } else if (score > max_noty_score) {
          max_noty_score = score;
          maxl = l;
        }
      }

      double loss = 1.0 - y_score + max_noty_score;
      if (loss > 0.0) {
        double tau = tau_calc(loss, 2.0 * x_vec[i].snorm());
        (*weight)[yl]->Add(x_vec[i], tau);
        (*weight)[maxl]->Add(x_vec[i], -1.0 * tau);
        ++miss_num;
      }
    }
    if (verbose && t % 10 == 0) { fprintf(stderr, "."); }
    if (miss_num == 0) { break; }
  }
  if (verbose) { fprintf(stderr, "done(%d)\n", t); }

  return 1;
}

} // namespace



namespace toybox {

class TauCalcPA {
  public:
    TauCalcPA() {};
    double operator()(double l, double sn) { return l / sn; };
};
class TauCalcPA1 {
  public:
    TauCalcPA1(double c) : c_(c) {};
    double operator()(double l, double sn) { return std::min(c_, l / sn); };
  private:
    TauCalcPA1();
    double c_;
};
class TauCalcPA2 {
  public:
    TauCalcPA2(double c) : inv2c_(1.0 / (2.0 * c)) {};
    double operator()(double l, double sn) { return l / (sn + inv2c_); };
  private:
    TauCalcPA2();
    double inv2c_;
};


PassiveAggressive::PassiveAggressive()
  : lnum_(0), fnum_(0), labels_(NULL), weight_() {
}

PassiveAggressive::~PassiveAggressive() {
  if (labels_ != NULL) {
    delete[] labels_; labels_ = NULL;
  }
  weight_.clear();
}

void PassiveAggressive::Train(const std::vector<SVector> &x_vec,
                       const std::vector<int> &y_vec,
                       const struct Parameter &param) {

  if (x_vec.size() != y_vec.size() || x_vec.size() == 0 || param.T <= 0) {
    return;
  }

  lnum_ = 0;
  fnum_ = 0;
  std::vector<int> labels;

  int dnum = x_vec.size();
  get_lnum_fnum_labels(x_vec, y_vec, &lnum_, &fnum_, &labels);
  
  labels_ = new int[lnum_];
  for (int l = 0; l < lnum_; ++l) {
    labels_[l] = labels[l];
  }

  const ModelType &mtype = param.model_type;
  
  if (mtype == BIN_PA  || mtype == BIN_PA1 || mtype == BIN_PA2) {
    int *suby = new int[dnum];
    WVector *w;
    int weight_num = (lnum_ == 2) ? 1 : lnum_;
    for (int l = 0; l < weight_num; ++l) {
      w = new WVector(fnum_);
      for (int i = 0; i < dnum; ++i) {
        if (y_vec[i] == labels_[l]) { suby[i] =  1; }
        else                        { suby[i] = -1; }
      }
      if (param.verbose) { fprintf(stderr, "%d ", l); }

      if (mtype == BIN_PA) {
        train_bin(param.T, x_vec, suby, param.verbose, TauCalcPA(), w);
      } else if (mtype == BIN_PA1) {
        train_bin(param.T, x_vec, suby, param.verbose, TauCalcPA1(param.C), w);
      } else if (mtype == BIN_PA2) {
        train_bin(param.T, x_vec, suby, param.verbose, TauCalcPA2(param.C), w);
      }
      weight_.push_back(w);
    }
    delete[] suby; suby = NULL;

  } else if (mtype == MC_PA  || mtype == MC_PA1 || mtype == MC_PA2) {
    for (int l = 0; l < lnum_; ++l) {
      weight_.push_back(new WVector(fnum_));
    }
    if (mtype == MC_PA) {
      train_multi(param.T, lnum_, fnum_, x_vec, y_vec,
                  labels_, param.verbose, TauCalcPA(), &weight_);
    } else if (mtype == MC_PA1) {
      train_multi(param.T, lnum_, fnum_, x_vec, y_vec,
                  labels_, param.verbose, TauCalcPA1(param.C), &weight_);
    } else if (mtype == MC_PA2) {
      train_multi(param.T, lnum_, fnum_, x_vec, y_vec,
                  labels_, param.verbose, TauCalcPA2(param.C), &weight_);
    }
  } else {
    fprintf(stderr, "Unknown model_type: %d\n", param.model_type);
  }

  return;
}

int PassiveAggressive::Predict(const SVector &x, double *predict) const {

  if (labels_ == NULL || weight_.size() == 0) { return -1; }

  int maxl = -1;
  if (lnum_ == 2) {
    double score = weight_[0]->Dot(x);
    maxl = (score >= 0.0) ? 0 : 1;
    if (predict != NULL) {
      predict[0] = score;
      predict[1] = -1 * score;
    }
  } else {
    double max_score = -DBL_MAX;
    for (int l = 0; l < lnum_; ++l) {
      double score = weight_[l]->Dot(x);
      if (score > max_score) {
        max_score = score;
        maxl = l;
      }
      if (predict != NULL) {
        predict[l] = score;
      }
    }
  }

  return maxl;
}

int PassiveAggressive::Save(const char *filename) const {
  FILE *fp = fopen(filename, "w");
  if (fp == NULL) { return -1; }

  fprintf(fp, "lnum %d\n", lnum_); 
  fprintf(fp, "fnum %d\n", fnum_); 

  fprintf(fp, "label");
  for (int l = 0; l < lnum_; ++l) {
    fprintf(fp, " %d", labels_[l]);
  }
  fprintf(fp, "\n");

  fprintf(fp, "weight\n");
  std::string str;
  for (size_t l = 0; l < weight_.size(); ++l) {
    str.clear();
    weight_[l]->ToString(&str);
    fprintf(fp, "%s\n", str.c_str());
  }

  fclose(fp);

  return 1;
}

int PassiveAggressive::Load(const char *filename) {
  if (labels_ != NULL) {
    delete[] labels_; labels_ = NULL;
  }
  if (weight_.size() > 0) {
    for (size_t i = 0; i < weight_.size(); ++i) {
      delete weight_[i]; weight_[i] = NULL;
    }
  }
  weight_.clear();

  std::ifstream ifs(filename, std::ifstream::in);
  std::string buff;

  while (ifs && std::getline(ifs, buff)) {
    if (buff.substr(0, 4) == "lnum") {
      lnum_ = static_cast<int>(strtol(buff.substr(5).c_str(), NULL, 10));
    } else if (buff.substr(0, 4) == "fnum") {
      fnum_ = static_cast<int>(strtol(buff.substr(5).c_str(), NULL, 10));
    } else if (buff.substr(0, 5) == "label") {
      if (lnum_ < 2) { return -1; }
      labels_ = new int[lnum_];
      const char *label_str = buff.substr(6).c_str();
      int length = strlen(label_str);
      char cbuff[length + 1];
      memmove(cbuff, label_str, length + 1);
      char *p = cbuff;
      for (int i = 0; i < lnum_; ++i) {
        char *cl = strtok(p, " ");
        if (cl == NULL) { break; }
        int l = static_cast<int>(strtol(cl, NULL, 10));
        labels_[i] = l;
        p = NULL;
      }
      if (lnum_ < 2) { return -1; }
    } else if (buff.substr(0, 1) == "w") {
      break;
    }
  }
  if (ifs.eof()) { return -1; }

  while (ifs && std::getline(ifs, buff)) {
    toybox::WVector *w = new toybox::WVector(buff.c_str());
    weight_.push_back(w);
  }
  int w_size = static_cast<int>(weight_.size());
  if (lnum_ == 2) {
    if (w_size != 1) { return -1; }
  } else {
    if (w_size != lnum_) { return -1; }
  }

  return 1;
}

int PassiveAggressive::GetLabels(int *labels) const {
  if (labels != NULL) {
    for (int l = 0; l < lnum_; ++l) {
      labels[l] = labels_[l];
    }
  }
  return lnum_;
}




} // namespace toybox
