// b70torch stand-in for <thrust/complex.h>. torch's complex headers use
// thrust::complex<T> as a conversion target and call a fixed set of math
// functions on it (see c10/util/complex_math.h); this provides exactly that
// with plain formulas usable in host and device code. Real Thrust (algorithms,
// device vectors) is not available through this header.
#pragma once
#include <cmath>
#include <complex>

namespace thrust {

template <typename T>
struct complex {
  T re = T(0), im = T(0);
  __host__ __device__ complex() = default;
  __host__ __device__ complex(T r, T i = T(0)) : re(r), im(i) {}
  template <typename U>
  __host__ __device__ complex(const complex<U> &o) : re(T(o.real())), im(T(o.imag())) {}
  complex(const std::complex<T> &o) : re(o.real()), im(o.imag()) {}
  __host__ __device__ T real() const { return re; }
  __host__ __device__ T imag() const { return im; }
  __host__ __device__ void real(T r) { re = r; }
  __host__ __device__ void imag(T i) { im = i; }
  operator std::complex<T>() const { return std::complex<T>(re, im); }
  __host__ __device__ complex &operator+=(const complex &o) { re += o.re; im += o.im; return *this; }
  __host__ __device__ complex &operator-=(const complex &o) { re -= o.re; im -= o.im; return *this; }
  __host__ __device__ complex &operator*=(const complex &o) {
    T r = re * o.re - im * o.im; im = re * o.im + im * o.re; re = r; return *this;
  }
  __host__ __device__ complex &operator*=(T s) { re *= s; im *= s; return *this; }
  __host__ __device__ complex &operator/=(const complex &o) {
    T d = o.re * o.re + o.im * o.im;
    T r = (re * o.re + im * o.im) / d; im = (im * o.re - re * o.im) / d; re = r; return *this;
  }
  __host__ __device__ complex &operator/=(T s) { re /= s; im /= s; return *this; }
};

template <typename T> __host__ __device__ complex<T> operator+(complex<T> a, const complex<T> &b) { return a += b; }
template <typename T> __host__ __device__ complex<T> operator-(complex<T> a, const complex<T> &b) { return a -= b; }
template <typename T> __host__ __device__ complex<T> operator*(complex<T> a, const complex<T> &b) { return a *= b; }
template <typename T> __host__ __device__ complex<T> operator*(complex<T> a, T s) { return a *= s; }
template <typename T> __host__ __device__ complex<T> operator*(T s, complex<T> a) { return a *= s; }
template <typename T> __host__ __device__ complex<T> operator/(complex<T> a, const complex<T> &b) { return a /= b; }
template <typename T> __host__ __device__ complex<T> operator/(complex<T> a, T s) { return a /= s; }
template <typename T> __host__ __device__ complex<T> operator-(const complex<T> &a) { return complex<T>(-a.re, -a.im); }
template <typename T> __host__ __device__ bool operator==(const complex<T> &a, const complex<T> &b) { return a.re == b.re && a.im == b.im; }
template <typename T> __host__ __device__ bool operator!=(const complex<T> &a, const complex<T> &b) { return !(a == b); }

template <typename T> __host__ __device__ T abs(const complex<T> &a) { return std::hypot(a.re, a.im); }
template <typename T> __host__ __device__ T arg(const complex<T> &a) { return std::atan2(a.im, a.re); }
template <typename T> __host__ __device__ T norm(const complex<T> &a) { return a.re * a.re + a.im * a.im; }
template <typename T> __host__ __device__ complex<T> conj(const complex<T> &a) { return complex<T>(a.re, -a.im); }
template <typename T> __host__ __device__ complex<T> proj(const complex<T> &a) {
  if (std::isinf(a.re) || std::isinf(a.im)) return complex<T>(INFINITY, std::copysign(T(0), a.im));
  return a;
}
template <typename T> __host__ __device__ complex<T> polar(const T &r, const T &theta = T()) {
  return complex<T>(r * std::cos(theta), r * std::sin(theta));
}
template <typename T> __host__ __device__ complex<T> exp(const complex<T> &z) {
  T e = std::exp(z.re);
  return complex<T>(e * std::cos(z.im), e * std::sin(z.im));
}
template <typename T> __host__ __device__ complex<T> log(const complex<T> &z) {
  return complex<T>(std::log(abs(z)), arg(z));
}
template <typename T> __host__ __device__ complex<T> log10(const complex<T> &z) { return log(z) / T(2.302585092994045684); }
template <typename T> __host__ __device__ complex<T> log2(const complex<T> &z) { return log(z) / T(0.6931471805599453094); }
template <typename T> __host__ __device__ complex<T> sqrt(const complex<T> &z) {
  T r = abs(z);
  if (r == T(0)) return complex<T>(T(0), z.im);
  T x = std::sqrt((r + std::fabs(z.re)) / T(2));
  if (z.re >= T(0)) return complex<T>(x, z.im / (T(2) * x));
  return complex<T>(std::fabs(z.im) / (T(2) * x), std::copysign(x, z.im));
}
template <typename T> __host__ __device__ complex<T> pow(const complex<T> &z, const complex<T> &w) {
  if (z.re == T(0) && z.im == T(0)) return (w.re == T(0) && w.im == T(0)) ? complex<T>(T(1)) : complex<T>(T(0));
  return exp(w * log(z));
}
template <typename T> __host__ __device__ complex<T> pow(const complex<T> &z, const T &y) { return pow(z, complex<T>(y)); }
template <typename T> __host__ __device__ complex<T> pow(const T &x, const complex<T> &w) { return pow(complex<T>(x), w); }
template <typename T> __host__ __device__ complex<T> sin(const complex<T> &z) {
  return complex<T>(std::sin(z.re) * std::cosh(z.im), std::cos(z.re) * std::sinh(z.im));
}
template <typename T> __host__ __device__ complex<T> cos(const complex<T> &z) {
  return complex<T>(std::cos(z.re) * std::cosh(z.im), -std::sin(z.re) * std::sinh(z.im));
}
template <typename T> __host__ __device__ complex<T> tan(const complex<T> &z) { return sin(z) / cos(z); }
template <typename T> __host__ __device__ complex<T> sinh(const complex<T> &z) {
  return complex<T>(std::sinh(z.re) * std::cos(z.im), std::cosh(z.re) * std::sin(z.im));
}
template <typename T> __host__ __device__ complex<T> cosh(const complex<T> &z) {
  return complex<T>(std::cosh(z.re) * std::cos(z.im), std::sinh(z.re) * std::sin(z.im));
}
template <typename T> __host__ __device__ complex<T> tanh(const complex<T> &z) { return sinh(z) / cosh(z); }
template <typename T> __host__ __device__ complex<T> asinh(const complex<T> &z) {
  return log(z + sqrt(z * z + complex<T>(T(1))));
}
template <typename T> __host__ __device__ complex<T> acosh(const complex<T> &z) {
  return log(z + sqrt(z + complex<T>(T(1))) * sqrt(z - complex<T>(T(1))));
}
template <typename T> __host__ __device__ complex<T> atanh(const complex<T> &z) {
  const complex<T> one(T(1));
  return (log(one + z) - log(one - z)) / T(2);
}
template <typename T> __host__ __device__ complex<T> asin(const complex<T> &z) {
  // asin(z) = -i * asinh(i z)
  complex<T> w = asinh(complex<T>(-z.im, z.re));
  return complex<T>(w.im, -w.re);
}
template <typename T> __host__ __device__ complex<T> acos(const complex<T> &z) {
  complex<T> s = asin(z);
  return complex<T>(T(1.5707963267948966192) - s.re, -s.im);
}
template <typename T> __host__ __device__ complex<T> atan(const complex<T> &z) {
  // atan(z) = -i * atanh(i z)
  complex<T> w = atanh(complex<T>(-z.im, z.re));
  return complex<T>(w.im, -w.re);
}

}  // namespace thrust
