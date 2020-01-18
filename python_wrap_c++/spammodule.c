#define PY_SSIZE_T_CLEAN  /* Make "s#" use Py_ssize_t rather than int*/
#include <Python.h>

/*自定义Exception*/
static PyObject *SpamError;

/*在module initialization function中初始化 exception object*/
PyMODINIT_FUNC
PyInit_spam(void)
{
    PyObject *m;

    m = PyModule_Create(&spammodule);   // where spammodule comes from
    if (m == NULL) {
        return NULL;
    }

    SpamError = PyErr_NewException("spam.error", NULL, NULL);
    Py_XINCREF(SpamError);
    if (PyModule_AddObject(m, "error", SpamError) < 0) {
        Py_XDECREF(SpamError);
        Py_CLEAR(SpamError);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}

/*通过 spam.error 使用自定义Exception*/

static PyObject *
spam_system(PyObject *self, PyObject *args)
/* self 指向上级对象，module object or object instance*/
{
    const char *command;
    int sts;

    // PyArg_ParseTuple(PyObject *args, const char *format, ...)  将PyObject转为C对象
    // args 是一个tuple对象
    // s template string 决定转换方式，返回一个string
    // s# 返回一个string 加 string size
    // 返回零失败
    if (!PyArg_ParseTuple(args, "s", &command))
        return NULL;
    sts = system(command);
    if (sts < 0) {
        // 使用上自定义的Exception
        PyErr_SetString(SpamError, "System command failed");
        return NULL;
    }
    // 将int对象转为PyObject对象
    return PyLong_FromLong(sts);
}


/*
 * 调用方式
 * >>> import spam
 * >>> status = spam.system("ls -l")
 * 参数会打包成一个tuple对象传入c扩展函数函数中
 */
