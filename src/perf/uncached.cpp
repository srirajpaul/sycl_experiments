
#ifndef UNCACHED_H
#define UNCACHED_H

#include <CL/sycl.hpp>

#define USE_BUILTIN 0

// ############################
// copied from https://github.com/intel/intel-graphics-compiler/blob/master/IGC/BiFModule/Implementation/IGCBiF_Intrinsics_Lsc.cl#L90
// Load message caching control
enum LSC_LDCC {
    LSC_LDCC_DEFAULT      = 0,
    LSC_LDCC_L1UC_L3UC    = 1,   // Override to L1 uncached and L3 uncached
    LSC_LDCC_L1UC_L3C     = 2,   // Override to L1 uncached and L3 cached
    LSC_LDCC_L1C_L3UC     = 3,   // Override to L1 cached and L3 uncached
    LSC_LDCC_L1C_L3C      = 4,   // Override to L1 cached and L3 cached
    LSC_LDCC_L1S_L3UC     = 5,   // Override to L1 streaming load and L3 uncached
    LSC_LDCC_L1S_L3C      = 6,   // Override to L1 streaming load and L3 cached
    LSC_LDCC_L1IAR_L3C    = 7,   // Override to L1 invalidate-after-read, and L3 cached
};
//#include <IGCBiF_Intrinsics_Lsc.cl>
//
enum LSC_STCC {
    LSC_STCC_DEFAULT      = 0,
    LSC_STCC_L1UC_L3UC    = 1,   // Override to L1 uncached and L3 uncached
    LSC_STCC_L1UC_L3WB    = 2,   // Override to L1 uncached and L3 written back
    LSC_STCC_L1WT_L3UC    = 3,   // Override to L1 written through and L3 uncached
    LSC_STCC_L1WT_L3WB    = 4,   // Override to L1 written through and L3 written back
    LSC_STCC_L1S_L3UC     = 5,   // Override to L1 streaming and L3 uncached
    LSC_STCC_L1S_L3WB     = 6,   // Override to L1 streaming and L3 written back
    LSC_STCC_L1WB_L3WB    = 7,   // Override to L1 written through and L3 written back
};

typedef sycl::vec<uint, 2> uint2;
typedef sycl::vec<uint, 3> uint3;
typedef sycl::vec<uint, 4> uint4;
typedef sycl::vec<uint, 8> uint8;
typedef sycl::vec<ulong, 2> ulong2;
typedef sycl::vec<ulong, 3> ulong3;
typedef sycl::vec<ulong, 4> ulong4;
typedef sycl::vec<ulong, 8> ulong8;

#ifdef __SYCL_DEVICE_ONLY__
#define __global 
///////////////////////////////////////////////////////////////////////
// LSC Loads
///////////////////////////////////////////////////////////////////////
// global address space gathering load
SYCL_EXTERNAL extern "C" uint    __builtin_IB_lsc_load_global_uint  (const __global uint   *base, int immElemOff, enum LSC_LDCC cacheOpt); //D32V1
SYCL_EXTERNAL  uint2   __builtin_IB_lsc_load_global_uint2 (const __global uint2  *base, int immElemOff, enum LSC_LDCC cacheOpt); //D32V2
SYCL_EXTERNAL  uint3   __builtin_IB_lsc_load_global_uint3 (const __global uint3  *base, int immElemOff, enum LSC_LDCC cacheOpt); //D32V3
SYCL_EXTERNAL  uint4   __builtin_IB_lsc_load_global_uint4 (const __global uint4  *base, int immElemOff, enum LSC_LDCC cacheOpt); //D32V4
SYCL_EXTERNAL  uint8   __builtin_IB_lsc_load_global_uint8 (const __global uint8  *base, int immElemOff, enum LSC_LDCC cacheOpt); //D32V8
SYCL_EXTERNAL extern "C" ulong   __builtin_IB_lsc_load_global_ulong (const __global ulong  *base, int immElemOff, enum LSC_LDCC cacheOpt); //D64V1
SYCL_EXTERNAL  ulong2  __builtin_IB_lsc_load_global_ulong2(const __global ulong2 *base, int immElemOff, enum LSC_LDCC cacheOpt); //D64V2
SYCL_EXTERNAL  ulong3  __builtin_IB_lsc_load_global_ulong3(const __global ulong3 *base, int immElemOff, enum LSC_LDCC cacheOpt); //D64V3
SYCL_EXTERNAL  ulong4  __builtin_IB_lsc_load_global_ulong4(const __global ulong4 *base, int immElemOff, enum LSC_LDCC cacheOpt); //D64V4
SYCL_EXTERNAL  ulong8  __builtin_IB_lsc_load_global_ulong8(const __global ulong8 *base, int immElemOff, enum LSC_LDCC cacheOpt); //D64V8

// global address space scattering store
SYCL_EXTERNAL extern "C" void  __builtin_IB_lsc_store_global_uint  (__global uint   *base, int immElemOff, uint   val, enum LSC_STCC cacheOpt); //D32V1
SYCL_EXTERNAL extern "C" void  __builtin_IB_lsc_store_global_uint2 (__global uint2  *base, int immElemOff, uint2  val, enum LSC_STCC cacheOpt); //D32V2
SYCL_EXTERNAL extern "C" void  __builtin_IB_lsc_store_global_uint3 (__global uint3  *base, int immElemOff, uint3  val, enum LSC_STCC cacheOpt); //D32V3
SYCL_EXTERNAL extern "C" void  __builtin_IB_lsc_store_global_uint4 (__global uint4  *base, int immElemOff, uint4  val, enum LSC_STCC cacheOpt); //D32V4
SYCL_EXTERNAL extern "C" void  __builtin_IB_lsc_store_global_uint8 (__global uint8  *base, int immElemOff, uint8  val, enum LSC_STCC cacheOpt); //D32V8
SYCL_EXTERNAL extern "C" void  __builtin_IB_lsc_store_global_ulong (__global ulong  *base, int immElemOff, ulong  val, enum LSC_STCC cacheOpt); //D64V1
SYCL_EXTERNAL extern "C" void  __builtin_IB_lsc_store_global_ulong2(__global ulong2 *base, int immElemOff, ulong2 val, enum LSC_STCC cacheOpt); //D64V2
SYCL_EXTERNAL extern "C" void  __builtin_IB_lsc_store_global_ulong3(__global ulong3 *base, int immElemOff, ulong3 val, enum LSC_STCC cacheOpt); //D64V3
SYCL_EXTERNAL extern "C" void  __builtin_IB_lsc_store_global_ulong4(__global ulong4 *base, int immElemOff, ulong4 val, enum LSC_STCC cacheOpt); //D64V4
SYCL_EXTERNAL extern "C" void  __builtin_IB_lsc_store_global_ulong8(__global ulong8 *base, int immElemOff, ulong8 val, enum LSC_STCC cacheOpt); //D64V8

#endif //! __SYCL_DEVICE_ONLY




static inline void ucs_uint(uint  *base, uint  val)
{
#ifdef __SYCL_DEVICE_ONLY__
#if USE_BUILTIN
  __builtin_IB_lsc_store_global_uint (base, 0, val, LSC_STCC_L1UC_L3UC);
#else
  *base = val;
#endif
#else
  *base = val;
#endif
}

static inline void ucs_ulong(ulong  *base, ulong  val)
{
#ifdef __SYCL_DEVICE_ONLY__
#if USE_BUILTIN
  __builtin_IB_lsc_store_global_ulong (base, 0, val, LSC_STCC_L1UC_L3UC);
#else
  *base = val;
#endif
#else
  *base = val;
#endif
}

static inline ulong ucl_ulong(ulong  *base)
{
  ulong v;
#ifdef __SYCL_DEVICE_ONLY__
#if USE_BUILTIN
  v = __builtin_IB_lsc_load_global_ulong (base, 0, LSC_LDCC_L1UC_L3UC);
#else
  v = *(( ulong *) base);
#endif
#else
  v = *base;
#endif
  return(v);
}

static inline void ucs_ulong2(ulong2  *base, ulong2  val)
{
#ifdef __SYCL_DEVICE_ONLY__
#if USE_BUILTIN
  __builtin_IB_lsc_store_global_ulong2 (base, 0, val, LSC_STCC_L1UC_L3UC);
#else
  *base = val;
#endif
#else
  *base = val;
#endif
}

static inline ulong2 ucl_ulong2(ulong2  *base)
{
  ulong2 v;
#ifdef __SYCL_DEVICE_ONLY__
#if USE_BUILTIN
  v = __builtin_IB_lsc_load_global_ulong2 (base, 0, LSC_LDCC_L1UC_L3UC);
#else
  v = *(( ulong2 *) base);
#endif
#else
  v = *base;
#endif
  return(v);
}

static inline void ucs_ulong4(ulong4  *base, ulong4  val)
{
#ifdef __SYCL_DEVICE_ONLY__
#if USE_BUILTIN
  __builtin_IB_lsc_store_global_ulong4 (base, 0, val, LSC_STCC_L1UC_L3UC);
#else
  *base = val;
#endif
#else
  *base = val;
#endif
}

static inline ulong4 ucl_ulong4(ulong4  *base)
{
  ulong4 v;
#ifdef __SYCL_DEVICE_ONLY__
#if USE_BUILTIN
  v = __builtin_IB_lsc_load_global_ulong4 (base, 0, LSC_LDCC_L1UC_L3UC);
#else
  v = *(( ulong4 *) base);
#endif
#else
  v = *base;
#endif
  return(v);
}


static inline void ucs_ulong8(ulong8  *base, ulong8  val)
{
#ifdef __SYCL_DEVICE_ONLY__
#if USE_BUILTIN
  __builtin_IB_lsc_store_global_ulong8 (base, 0, val, LSC_STCC_L1UC_L3UC);
#else
  *base = val;
#endif
#else
  *base = val;
#endif
}

static inline ulong8 ucl_ulong8(ulong8  *base)
{
  ulong8 v;
#ifdef __SYCL_DEVICE_ONLY__
#if USE_BUILTIN
  v = __builtin_IB_lsc_load_global_ulong8 (base, 0, LSC_LDCC_L1UC_L3UC);
#else
  v = *(( ulong8 *) base);
#endif
#else
  v = *base;
#endif
  return(v);
}


#endif //! ifndef UNCACHED_H
