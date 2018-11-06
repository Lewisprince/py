class High_school_student():
    def __init__(self,age,sex): # the underline around 'init' are two!
        self.age=age
        self.sex=sex
    def missing_detecting(self):
        if self.age=='':
            print("missing")
        else:
            print("not missing")
s1=High_school_student('17','M')
s2=High_school_student('','F')

s1.missing_detecting()
s2.missing_detecting()
print(s1.age)
