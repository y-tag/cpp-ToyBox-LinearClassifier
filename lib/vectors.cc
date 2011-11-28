#include "vectors.h"

#include <cstdio>
#include <cassert>
#include <cstring>
#include <cmath>

#include <algorithm>

static const int min_capacity = 16;
static const int max_expand   = 4096;

namespace {

int split_to_pairs(const char *str,
                   std::vector<std::pair<int, double> > *pvec) {
  if (str == NULL || pvec == NULL) { return -1; }
  int length = strlen(str);
  char cbuff[length + 1];
  memmove(cbuff, str, length + 1);

  char *p = cbuff;
  while (1) {
    char *f = strtok(p, ":");
    char *v = strtok(NULL, " \t");
    if (v == NULL) {
      break;
    }
    pvec->push_back(std::make_pair(static_cast<int>(strtol(f, NULL, 10)),
                    strtod(v, NULL)));
    p = NULL;
  }
  if (p != NULL) { return 0; }
  return 1;
}

toybox::SVector::Pair* search(toybox::SVector::Pair *pairs, int num_pairs, int i) {
  int low  = 0;
  int high = num_pairs - 1;
  while (high >= low) {
    int mid = (high + low) / 2;
    if (pairs[mid].index > i) {
      high = mid - 1;
    } else if (pairs[mid].index < i) {
      low  = mid + 1;
    } else {
      return &pairs[mid];
    }
  }
  return NULL;
}

} // namespace

namespace toybox {

FVector::FVector() : values_(NULL), scale_(1.0), size_(0), snorm_(0.0) {}

FVector::FVector(int n) : scale_(1.0), size_(n), snorm_(0.0) {
  values_ = new VFloat[n];
  std::fill_n(values_, n, 0.0);
}

FVector::FVector(const SVector &svec)
 : scale_(svec.scale_), size_(svec.size_), snorm_(-1.0) {
  values_ = new VFloat[svec.size_];
  std::fill_n(values_, svec.size_, 0.0);
  snorm_ = 0.0;
  for (int i = 0; svec.pairs_[i].index != -1; ++i) {
    values_[svec.pairs_[i].index] = svec.pairs_[i].value;
    snorm_ += static_cast<double>(svec.pairs_[i].value * svec.pairs_[i].value);
  }
}

FVector::FVector(double *array, int n)
 : scale_(1.0), size_(n), snorm_(-1.0) {
  values_ = new VFloat[n];
  snorm_ = 0.0;
  for (int i = 0; i < n; ++i) {
    values_[i] = static_cast<VFloat>(array[i]);
    snorm_ += array[i] * array[i];
  }
}

FVector::FVector(const std::vector<double> &vec)
 : scale_(1.0), size_(vec.size()), snorm_(-1.0) {
  values_ = new VFloat[vec.size()];
  snorm_ = 0.0;
  for (size_t i = 0; i < vec.size(); ++i) {
    values_[i] = static_cast<VFloat>(vec[i]);
    snorm_ += vec[i] * vec[i];
  }
}

FVector::FVector(const std::vector<std::pair<int, double> > &pvec)
  : scale_(1.0), snorm_(-1.0) {
  int pvec_size = pvec.size();
  size_ = pvec[pvec_size - 1].first + 1;
  values_ = new VFloat[size_];
  std::fill_n(values_, size_, 0.0);
  snorm_ = 0.0;
  for (int i = 0; i < pvec_size; ++i) {
    values_[pvec[i].first] = static_cast<VFloat>(pvec[i].second);
    snorm_ += pvec[i].second * pvec[i].second;
  }
}

FVector::FVector(const char *str) : scale_(1.0), snorm_(-1.0) {
  std::vector<std::pair<int, double> > pvec;
  split_to_pairs(str, &pvec);
  int pvec_size = pvec.size();
  size_ = pvec[pvec_size - 1].first + 1;
  values_ = new VFloat[size_];
  std::fill_n(values_, size_, 0.0);
  snorm_ = 0.0;
  for (int i = 0; i < pvec_size; ++i) {
    values_[pvec[i].first] = static_cast<VFloat>(pvec[i].second);
    snorm_ += pvec[i].second * pvec[i].second;
  }
}

FVector::~FVector() {
  if (values_ != NULL) {
    delete[] values_; values_ = NULL;
  }
}

double FVector::Get(int i) const {
  if (size_ > i) {
    return static_cast<double>(values_[i]) * scale_;
  } else {
    return 0.0;
  }
}

void FVector::Set(int i, double v) {
  assert(size_ > i);
  snorm_ -= static_cast<double>(values_[i] * values_[i]);
  values_[i] = static_cast<VFloat>(v / scale_);
  snorm_ += v * v;
}

void FVector::Add(const SVector &svec, double c) {
  int svec_size = svec.size();
  if (svec_size > size_) {
    VFloat *new_values = new VFloat[svec_size];
    memcpy(new_values, values_, size_);
    std::fill_n(&(new_values[size_]), svec_size - size_, 0.0);
    delete[] values_;
    values_ = new_values;
    size_   = svec_size;
  }

  double sfrac = svec.scale_ / scale_;
  for (int i = 0; svec.pairs_[i].index != -1; ++i) {
    int idx = svec.pairs_[i].index;
    snorm_ -= static_cast<double>(values_[idx] * scale_ *
                                  values_[idx] * scale_);
    values_[svec.pairs_[i].index]
      += static_cast<VFloat>(c * svec.pairs_[i].value * sfrac);
    snorm_ += static_cast<double>(values_[idx] * scale_ *
                                  values_[idx] * scale_);
  }
  return;
}

double FVector::Dot(const SVector &svec) const {
  double prod = 0.0;
  for (int i = 0; svec.pairs_[i].index != -1; ++i) {
    prod += Get(svec.pairs_[i].index) * svec.scale_ * svec.pairs_[i].value;
  }
  return prod;
}

double FVector::Dot(const FVector &fvec) const {
  double prod = 0.0;
  int min_size = (size_ > fvec.size()) ? fvec.size() : size_;
  for (int i = 0; i < min_size; ++i) {
    prod += Get(i) * fvec.Get(i);
  }
  return prod;
}

void FVector::Scale(double c) {
  scale_ *= c;
  snorm_ *= fabs(c) * fabs(c);
  if (scale_ < 1e-10) {
    for (int i = 0; i < size_; ++i) {
      values_[i] *= scale_;
    }
    scale_ = 1.0;
  }
  return;
}

void FVector::Show() const {
  for (int i = 0; i < size_; ++i) {
    if (values_[i] != 0.0) {
      fprintf(stdout, "%d:%.16g ", i, scale_ * values_[i]);
    }
  }
  fprintf(stdout, "\n");
}

void FVector::ToString(std::string *str) const {
  if (str == NULL) { return; }
  char cbuff[256];
  for (int i = 0; i < size_; ++i) {
    if (values_[i] != 0.0) {
      if (! str->empty()) { *str += " "; }
      sprintf(cbuff, "%d:%.16g", i, scale_ * values_[i]);
      *str += cbuff;
    }
  }
  return;
}


SVector::SVector()
  : num_pairs_(0), scale_(1.0), size_(0), capacity_(min_capacity), snorm_(0.0) {
  pairs_ = new Pair[capacity_];
  pairs_[0].index = -1;
}

SVector::SVector(int)
  : num_pairs_(0), scale_(1.0), size_(0), capacity_(min_capacity), snorm_(0.0) {
  pairs_ = new Pair[capacity_];
  pairs_[0].index = -1;
}

SVector::SVector(const SVector &svec)
  : num_pairs_(svec.num_pairs_), scale_(svec.scale_),
    size_(svec.size_), capacity_(svec.capacity_), snorm_(-1.0) {
  pairs_ = new Pair[capacity_];
  snorm_ = 0.0;
  for (int i = 0; i < svec.num_pairs_; ++i) {
    pairs_[i].index = svec.pairs_[i].index;
    pairs_[i].value = svec.pairs_[i].value;
    snorm_ += static_cast<double>(pairs_[i].value * pairs_[i].value);
  }
  pairs_[svec.num_pairs_].index = -1;
}

SVector::SVector(const FVector &fvec) : scale_(1.0), snorm_(-1.0) {
  int num_nonzero = 0;
  int max_nonzero = 0;
  for (int i = 0; i < fvec.size(); ++i) {
    if (fvec.Get(i) != 0.0) {
      ++num_nonzero;
      max_nonzero = i;
    }
  }
  capacity_ = (num_nonzero + 1 > min_capacity) ? num_nonzero + 1 : min_capacity;
  size_ = max_nonzero + 1;
  pairs_ = new Pair[capacity_];
  snorm_ = 0.0;
  int j = 0;
  for (int i = 0; i < fvec.size(); ++i) {
    if (fvec.Get(i) != 0.0) {
      pairs_[j].index = i;
      pairs_[j].value = static_cast<VFloat>(fvec.Get(i));
      ++j;
      snorm_ += static_cast<double>(pairs_[i].value * pairs_[i].value);
    }
  }
  pairs_[j].index = -1;
  num_pairs_ = j;
}

SVector::SVector(double *array, int n) : scale_(1.0), snorm_(-1.0) {
  int num_nonzero = 0;
  int max_nonzero = 0;
  for (int i = 0; i < n; ++i) {
    if (array[i] != 0.0) {
      ++num_nonzero;
      max_nonzero = i;
    }
  }
  capacity_ = (num_nonzero + 1 > min_capacity) ? num_nonzero + 1 : min_capacity;
  size_ = max_nonzero + 1; pairs_ = new Pair[capacity_];
  snorm_ = 0.0;
  int j = 0;
  for (int i = 0; i < n; ++i) {
    if (array[i] != 0.0) {
      pairs_[j].index = i;
      pairs_[j].value = static_cast<VFloat>(array[i]);
      ++j;
      snorm_ += static_cast<double>(pairs_[i].value * pairs_[i].value);
    }
  }
  pairs_[j].index = -1;
  num_pairs_ = j;
}

SVector::SVector(const std::vector<double> &vec) : scale_(1.0), snorm_(-1.0) {
  int num_nonzero = 0;
  int max_nonzero = 0;
  for (size_t i = 0; i < vec.size(); ++i) {
    if (vec[i] != 0.0) {
      ++num_nonzero;
      max_nonzero = i;
    }
  }
  capacity_ = (num_nonzero + 1 > min_capacity) ? num_nonzero + 1 : min_capacity;
  size_ = max_nonzero + 1;
  pairs_ = new Pair[capacity_];
  snorm_ = 0.0;
  int j = 0;
  for (size_t i = 0; i < vec.size(); ++i) {
    if (vec[i] != 0.0) {
      pairs_[j].index = i;
      pairs_[j].value = static_cast<VFloat>(vec[i]);
      ++j;
      snorm_ += static_cast<double>(pairs_[i].value * pairs_[i].value);
    }
  }
  pairs_[j].index = -1;
  num_pairs_ = j;
}

SVector::SVector(const std::vector<std::pair<int, double> > &pvec)
  : scale_(1.0), snorm_(-1.0) {
  int pvec_size = pvec.size();
  capacity_ = (pvec_size + 1 > min_capacity) ? pvec_size + 1 : min_capacity;
  pairs_ = new Pair[capacity_];
  snorm_ = 0.0;
  int i;
  for (i = 0; i < pvec_size; ++i) {
    pairs_[i].index = pvec[i].first;
    pairs_[i].value = static_cast<VFloat>(pvec[i].second);
    snorm_ += static_cast<double>(pairs_[i].value * pairs_[i].value);
  }
  size_ = pvec[i - 1].first + 1;
  pairs_[pvec_size].index = -1;
  num_pairs_ = pvec_size;
}

SVector::SVector(const char *str) : scale_(1.0), snorm_(-1.0) {
  std::vector<std::pair<int, double> > pvec;
  split_to_pairs(str, &pvec);
  int pvec_size = pvec.size();
  capacity_ = (pvec_size + 1 > min_capacity) ? pvec_size + 1 : min_capacity;
  pairs_ = new Pair[capacity_];
  snorm_ = 0.0;
  int i;
  for (i = 0; i < pvec_size; ++i) {
    pairs_[i].index = pvec[i].first;
    pairs_[i].value = static_cast<VFloat>(pvec[i].second);
    snorm_ += static_cast<double>(pairs_[i].value * pairs_[i].value);
  }
  size_ = pvec[i - 1].first + 1;
  pairs_[pvec_size].index = -1;
  num_pairs_ = pvec_size;
}

SVector::~SVector() {
  if (pairs_ != NULL) {
    delete[] pairs_; pairs_ = NULL;
  }
}

double SVector::Get(int i) const {
  const SVector::Pair *p = search(pairs_, num_pairs_, i);
  if (p != NULL) {
    return static_cast<double>(p->value);
  } else {
    return 0.0;
  }
}

void SVector::Set(int i, double v) {
  assert(i >= 0);
  SVector::Pair *p = search(pairs_, num_pairs_, i);
  if (p != NULL) {
    snorm_ -= static_cast<double>(p->value * p->value);
    p->value = static_cast<VFloat>(v);
    snorm_ += v * v;
    return;
  }
  if (num_pairs_ >= capacity_ - 1) {
    capacity_ += (capacity_ > max_expand) ? max_expand : capacity_;
    Pair *new_pairs = new Pair[capacity_];
    for (int j = 0; j < num_pairs_; ++j) {
      new_pairs[j].index = pairs_[j].index;
      new_pairs[j].value = pairs_[j].value;
    }
    new_pairs[num_pairs_].index = -1;
    delete[] pairs_;
    pairs_ = new_pairs;
  }
  pairs_[num_pairs_ + 1].index = -1; 
  int j;
  for (j = num_pairs_; pairs_[j].index > i; --j) {
    pairs_[j].index = pairs_[j - 1].index;
    pairs_[j].value = pairs_[j - 1].value;
  }
  pairs_[j].index = i;
  pairs_[j].value = v;
  ++num_pairs_;
  if (i > size_) { size_ = i; }
  snorm_ += v * v;
  return;
}

void SVector::Add(const SVector &svec, double c) {
  const Pair *p1 = pairs_;
  const Pair *p2 = svec.pairs_;
  int num_union =0;
  int i1 = p1->index;
  int i2 = p2->index;
  while (i1 != -1 && i2 != -1) {
    if      (i1 < i2) { ++p1; i1 = p1->index; }
    else if (i1 > i2) { ++p2; i2 = p2->index; }
    else              { ++p1; i1 = p1->index; ++p2; i2 = p2->index; }
    ++num_union;
  }
  while (p1->index != -1) { ++p1; ++num_union; }
  while (p2->index != -1) { ++p2; ++num_union; }
  p1 = pairs_ + num_pairs_ - 1;
  p2 = svec.pairs_ + svec.num_pairs_ - 1;

  size_ = (size_ > svec.size_) ? size_ : svec.size_;

  Pair *tmp_pairs = NULL;
  Pair *tp = NULL;
  int tmp_cap = 0;
  if (num_union + 1 > capacity_) {
    tmp_cap = (min_capacity > num_union + 1) ? min_capacity : num_union + 1;
    tmp_pairs = new Pair[tmp_cap];
    tp = tmp_pairs + num_union;
  } else {
    tp = pairs_ + num_union;
  }
  tp->index = -1; --tp;

  snorm_ = 0.0;
  double sfrac = svec.scale_ / scale_;
  double tmp = 0.0;
  while (p1 >= pairs_ && p2 >= svec.pairs_) {
    if (p1->index < p2->index) {
      tp->index = p2->index;
      tp->value = static_cast<VFloat>(c * p2->value * sfrac);
      tmp = p2->value * svec.scale_ * c; snorm_ += tmp * tmp;
      --p2;
    } else if (p1->index > p2->index) {
      tp->index = p1->index;
      tp->value = p1->value;
      tmp = p1->value * scale_; snorm_ += tmp * tmp;
      --p1;
    } else {
      tp->index = p1->index;
      tp->value = static_cast<VFloat>(p1->value + c * p2->value * sfrac);
      tmp = p1->value * scale_; snorm_ += tmp * tmp;
      tmp = p2->value * svec.scale_ * c; snorm_ += tmp * tmp;
      --p1; --p2;
    }
    --tp;
  }
  while (p1 >= pairs_) {
    tp->index = p1->index;
    tp->value = p1->value;
    tmp = p1->value * scale_; snorm_ += tmp * tmp;
    --p1; --tp;
  }
  while (p2 >= svec.pairs_) {
    tp->index = p2->index;
    tp->value = static_cast<VFloat>(c * p2->value * sfrac);
    tmp = p2->value * svec.scale_ * c; snorm_ += tmp * tmp;
    --p2; --tp;
  }
  
  if (tmp_pairs != NULL) {
    delete[] pairs_;
    pairs_ = tmp_pairs;
    capacity_ = tmp_cap;
  }
  num_pairs_ = num_union;

  return;
}

double SVector::Dot(const SVector &svec) const {
  double prod = 0.0;
  const Pair *p1 = pairs_;
  const Pair *p2 = svec.pairs_;
  int i1 = p1->index;
  int i2 = p2->index;
  while (i1 != -1 && i2 != -1) {
    if (i1 < i2) {
      ++p1; i1 = p1->index;
    } else if (i1 > i2) {
      ++p2; i2 = p2->index;
    } else {
      prod += static_cast<VFloat>(scale_ * p1->value * svec.scale_ * p2->value);
      ++p1; i1 = p1->index;
      ++p2; i2 = p2->index;
    }
  }
  return prod;
}

void SVector::Scale(double c) {
  scale_ *= c;
  snorm_ *= fabs(c) * fabs(c);
  if (scale_ < 1e-10) {
    for (int i = 0; i < num_pairs_ - 1; ++i) {
      pairs_[i].value = static_cast<VFloat>(scale_ * pairs_[i].value);
    }
    scale_ = 1.0;
  }
  return;
}

void SVector::Show() const {
  for (int i = 0; i < num_pairs_; ++i) {
    fprintf(stdout, "%d:%.16g ", pairs_[i].index, scale_ * pairs_[i].value);
  }
  fprintf(stdout, "\n");
}

void SVector::ToString(std::string *str) const {
  if (str == NULL) { return; }
  char cbuff[256];
  for (const Pair *p = pairs_; p->index != -1; ++p) {
    if (p->value == 0.0) { continue; }
    if (! str->empty())  { *str += " "; }
    sprintf(cbuff, "%d:%.16g", p->index, scale_ * p->value);
    *str += cbuff;
  }
  return;
}



} // namespace toybox
