import sys
sys.path.insert(1, "/Users/shenming/personal/github/learning-by-practice/python_wrap_c++/build/lib.macosx-10.9-x86_64-3.7/")
import keywdarg

if __name__ == "__main__":
    keywdarg.parrot(18, state = "abc", action ="bcd", type="efg")
    keywdarg.parrot(18, "abc", "bcd", "efg")
    keywdarg.parrot(voltage = 18, state = "abc", action ="bcd", type="efg")
