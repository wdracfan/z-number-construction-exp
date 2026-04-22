class FS:
    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def membership_function(self):
        def mf(x):
            if x <= self.a or x >= self.d:
                return 0
            if self.b <= x <= self.c:
                return 1
            if self.a <= x <= self.b:
                return (x - self.a) / (self.b - self.a)
            if self.c <= x <= self.d:
                return (self.d - x) / (self.d - self.c)
        return mf
    
    def specificity(self, u: float) -> float:
        return 1 - (self.c + self.d - self.a - self.b) / (2 * u)
    
    def to_latex_string(self) -> str:
        return f'$\\FS({self.a}; {self.b}; {self.c}; {self.d})$'
    
    def __str__(self) -> str:
        return f'FS({self.a:.3f}, {self.b:.3f}, {self.c:.3f}, {self.d:.3f})'