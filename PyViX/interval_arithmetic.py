def add_int(a, b):
    # a and b are intervals
    if isinstance(a, Interval) and isinstance(b, Interval):
        return Interval(a.lb + b.lb, a.ub + b.ub)
    
    # a is a number and b is an interval
    if isinstance(a, (int, float)) and isinstance(b, Interval):
        return Interval(a + b.lb, a + b.ub)
    
    # a is an interval and b is a number
    if isinstance(a, Interval) and isinstance(b, (int, float)):
        return Interval(a.lb + b, a.ub + b)

def sub_int(a, b):
    # a and b are intervals
    if isinstance(a, Interval) and isinstance(b, Interval):
        return Interval(a.lb - b.ub, a.ub - b.lb)
    
    # a is a number and b is an interval
    if isinstance(a, (int, float)) and isinstance(b, Interval):
        return Interval(a - b.ub, a - b.lb)
    
    # a is an interval and b is a number
    if isinstance(a, Interval) and isinstance(b, (int, float)):
        return Interval(a.lb - b, a.ub - b)

def mul_int(a, b):
    # a and b are intervals
    if isinstance(a, Interval) and isinstance(b, Interval):
        return Interval(min(a.lb * b.lb, a.lb * b.ub, a.ub * b.lb, a.ub * b.ub), max(a.lb * b.lb, a.lb * b.ub, a.ub * b.lb, a.ub * b.ub))
    
    # a is a number and b is an interval
    if isinstance(a, (int, float)) and isinstance(b, Interval):
        return Interval(min(a * b.lb, a * b.ub), max(a * b.lb, a * b.ub))
    
    # a is an interval and b is a number
    if isinstance(a, Interval) and isinstance(b, (int, float)):
        return Interval(min(a.lb * b, a.ub * b), max(a.lb * b, a.ub * b))

def div_int(a, b):
    # a and b are intervals
    if isinstance(a, Interval) and isinstance(b, Interval):

        if b.lb <= 0 and b.ub >= 0:
            return Interval(-float('inf'), float('inf'))
        
        return Interval(min(a.lb / b.lb, a.lb / b.ub, a.ub / b.lb, a.ub / b.ub), max(a.lb / b.lb, a.lb / b.ub, a.ub / b.lb, a.ub / b.ub))

    # a is a number and b is an interval
    if isinstance(a, (int, float)) and isinstance(b, Interval):
            
        if b.lb <= 0 and b.ub >= 0:
            return Interval(-float('inf'), float('inf'))
        
        return Interval(min(a / b.lb, a / b.ub), max(a / b.lb, a / b.ub))
    
    # a is an interval and b is a number
    if isinstance(a, Interval) and isinstance(b, (int, float)):
            
        if b == 0:
            return Interval(-float('inf'), float('inf'))
        
        return Interval(min(a.lb / b, a.ub / b), max(a.lb / b, a.ub / b))



class Interval:

    lb = None
    ub = None

    def __init__(self, lb, ub):
        self.lb = lb
        self.ub = ub

        if lb > ub:
            raise Exception("Lower bound is greater than upper bound")

    def __repr__(self) -> str:
        return f"[{self.lb}, {self.ub}]"

    
    def __add__(self, other):
        return add_int(self, other)
    
    def __radd__(self, other):
        return add_int(self, other)
    
    def __sub__(self, other):
        return sub_int(self, other)
    
    def __rsub__(self, other):
        return sub_int(other, self)
    
    def __mul__(self, other):
        if isinstance(other, Interval):
            return mul_int(self, other)
        else:
            return mul_int(self, Interval(other, other))
        # return mul_int(self, other)
    
    __rmul__ = __mul__
    # def __rmul__(self, other):
    #     return mul_int(self, other)
    
    def __truediv__(self, other):
        return div_int(self, other)
    
    def __rtruediv__(self, other):
        return div_int(other, self)
    

if __name__ == "__main__":
    a = Interval(1, 2)
    b = Interval(3, 4)
    print(a+b)