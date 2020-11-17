#pragma once

// LEXIC: set of macros for getting the site index
#define LEXIC(it,iz,iy,ix,L) ( (it)*L[0]*L[1]*L[2] + (iz)*L[0]*L[1] + (iy)*L[0] + (ix) )
#define LEXIC_TZY(it,iz,iy,L) ( (it)*L[1]*L[2] + (iz)*L[1] + (iy) )
#define LEXIC_TZX(it,iz,ix,L) ( (it)*L[0]*L[2] + (iz)*L[0] + (ix) )
#define LEXIC_TYX(it,iy,ix,L) ( (it)*L[0]*L[1] + (iy)*L[0] + (ix) )
#define LEXIC_ZYX(iz,iy,ix,L) ( (iz)*L[0]*L[1] + (iy)*L[0] + (ix) )
#define LEXIC_TZ(it,iz,L) ( (it)*L[2] + (iz) )
#define	LEXIC_TY(it,iy,L) ( (it)*L[1] + (iy) )
#define	LEXIC_TX(it,ix,L) ( (it)*L[0] + (ix) )
#define	LEXIC_ZY(iz,iy,L) ( (iz)*L[1] + (iy) )
#define	LEXIC_ZX(iz,ix,L) ( (iz)*L[0] + (ix) )
#define	LEXIC_YX(iy,ix,L) ( (iy)*L[0] + (ix) )

#define BOOST_PP_VARIADICS 1
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/control/if.hpp>
#include <boost/preprocessor/tuple/to_seq.hpp>
#include <boost/preprocessor/arithmetic/mul.hpp>
#include <boost/preprocessor/facilities/overload.hpp>
#include <boost/preprocessor/facilities/empty.hpp>

// IS_EMPTY: Macro which says if __VA_ARGS__ is empty
// NOTE: based on the: http://gustedt.wordpress.com/2010/06/08/detect-empty-macro-arguments
#define __ARG16(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, ...) _15
#define __HAS_COMMA(...) __ARG16(__VA_ARGS__, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0)
#define __TRIGGER_PARENTHESIS_(...) ,
#define __PASTE5(_0, _1, _2, _3, _4) _0 ## _1 ## _2 ## _3 ## _4
#define __IS_EMPTY_CASE_0001 ,
#define __IS_EMPTY(_0, _1, _2, _3) __HAS_COMMA(__PASTE5(__IS_EMPTY_CASE_, _0, _1, _2, _3))

#define IS_EMPTY(...)			\
  __IS_EMPTY(								\
	     /* test if there is just one argument, eventually an empty one */ \
	     __HAS_COMMA(__VA_ARGS__),					\
	     /* test if _TRIGGER_PARENTHESIS_ together with the argument adds a comma */ \
	     __HAS_COMMA(__TRIGGER_PARENTHESIS_ __VA_ARGS__),		\
	     /* test if the argument together with a parenthesis adds a comma */ \
	     __HAS_COMMA(__VA_ARGS__ (/*empty*/)),			\
	     /* test if placing it between _TRIGGER_PARENTHESIS_ and the parenthesis adds a comma */ \
	     __HAS_COMMA(__TRIGGER_PARENTHESIS_ __VA_ARGS__ (/*empty*/)))


// NOTHING: Macro that does nothing
#define NOTHING(...)


// PARENTHESES: Macro that adds parentheses [*] around each element of __VA_ARGS__.
//              If use_value is 0 then [0] is used for each element of __VA_ARGS__.
#define __ADD_PARENTHESES(r, use_value, elem) BOOST_PP_IF( use_value,  [elem], [0])
#define __FOR_EACH_PAR(use_value, ...) BOOST_PP_SEQ_FOR_EACH(__ADD_PARENTHESES, use_value, \
							     BOOST_PP_TUPLE_TO_SEQ((__VA_ARGS__)))
#define PARENTHESES(use_value,...) BOOST_PP_IF(IS_EMPTY(__VA_ARGS__),	\
					       NOTHING,			\
					       __FOR_EACH_PAR) (use_value, __VA_ARGS__)

// PRODUCT: Macro that does the product of all __VA_ARGS__.
#define __MUL_1(a) (a)
#define __MUL_2(a,b) (a)*(b)
#define __MUL_3(a,b,c) __MUL_2(__MUL_2(a,b),c)
#define __MUL_4(a,b,c,d) __MUL_3(__MUL_2(a,b),c,d)
#define __MUL_5(a,b,c,d,e) __MUL_4(__MUL_2(a,b),c,d,e)
#define __MUL_6(a,b,c,d,e,f) __MUL_5(__MUL_2(a,b),c,d,e,f)
#define __MUL_7(a,b,c,d,e,f,g) __MUL_6(__MUL_2(a,b),c,d,e,f,g)
#define __MUL_8(a,b,c,d,e,f,g,h) __MUL_7(__MUL_2(a,b),c,d,e,f,g,h)
// define more if needed
#define __ONE(...) 1
#define __MUL(...) BOOST_PP_OVERLOAD(__MUL_,__VA_ARGS__)(__VA_ARGS__)
#define PRODUCT(...) BOOST_PP_IF(IS_EMPTY(__VA_ARGS__),			\
				 __ONE,					\
				 __MUL) (__VA_ARGS__)

// EQUAL: Add an equal in front of __VA_ARGS__. Equivalent to __VA_OPT__(=)
#define __EQUAL(...) =  __VA_ARGS__
#define EQUAL(...) BOOST_PP_IF(IS_EMPTY(__VA_ARGS__),		\
			       NOTHING,				\
			       __EQUAL) (__VA_ARGS__)


// Use the following for accessing off-diagonal terms of matrices with only off-diagonal stored
#define _OFF2(i,j) ((i*(i-1))/2 + j)
#define OFF2(i,j) (i>j ? _OFF2(i,j) : _OFF2(j,i))

#define _OFF3(i,j,k) ((i*(i-1)*(i-2))/6 + (j*(j-1))/2 + k)
#define OFF3(i,j,k) (i>j && j>k ? _OFF3(i,j,k) : \
		    (i>k && k>j ? _OFF3(i,k,j) : \
		    (j>i && i>k ? _OFF3(j,i,k) : \
		    (j>k && k>i ? _OFF3(j,k,i) : \
		    (k>i && i>j ? _OFF3(k,i,j) : \
		                  _OFF3(k,j,i))))))

