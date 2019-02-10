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

