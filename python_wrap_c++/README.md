reference: https://docs.python.org/zh-cn/3.7/extending/extending.html

传统上，如果一个模块叫 spam，则对应实现它的 C 文件叫 spammodule.c；如果这个模块名字非常长，比如 spammify，则这个模块的文件可以直接叫 spammify.c。

`Python.h` 中定义的所有用户可见的符号都是以`Py`或者`PY`开头的。

Exceptions are stored in a static global variable inside the interpreter.<br>
A second global variable stores the "associate value" of the exception (the second argument to `raise`)<br>
A third variable contains the stack traceback.

`PyErr_SetString` 设置 "associate value"

如果一个函数不需要返回任何对象
```
Py_INCREF(Py_None);
return Py_None;
```

A C extension for CPython is a shared library (e.g. a `.so` file on Linux, `.pyd` on Windows).

通过`distutils`即可building C或者C++ Extension

使用`keywdarg.c`的例子的确能够生成
