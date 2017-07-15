a = set(["jake", "john", "eric", "john"])
print(a)


class Employee(object):
    """Models real-life employees!"""

    def __init__(self, employee_name):
        self.employee_name = employee_name

    def calculate_wage(self, hours):
        self.hours = hours
        return hours * 20.00


# Add your code below!
class PartTimeEmployee(Employee):
    def calculate_wage(self, hours):
        self.hours = hours
        return hours * 12.00

    def full_time_wage(self, hours):
        return super(PartTimeEmployee, self).calculate_wage(hours)


milton = PartTimeEmployee("milton")
milton.hours = 10.00
x = milton.full_time_wage(10.00)
print(x)


class Triangle(object):
    number_of_sides = 3

    def __init__(self, angle1, angle2, angle3):
        self.angle1 = angle1
        self.angle2 = angle2
        self.angle3 = angle3

    def check_angles(self):
        # return (sum(self.angle1, self.angle2, self.angle3) == 180)
        if self.angle1 + self.angle2 + self.angle3 == 180:
            return True
        else:
            return False


my_triangle = Triangle(30, 60, 90)
print(my_triangle.number_of_sides)
print(my_triangle.check_angles())


class Equilateral(Triangle):
    angle = 60

    def __init__(self):
        self.angle1 = self.angle
        self.angle2 = self.angle
        self.angle3 = self.angle


class Car(object):
    condition = "new"

    def __init__(self, model, color, mpg):
        self.model = model
        self.color = color
        self.mpg   = mpg

    def display_car(self):
        return "This is a %s %s with %d MPG." % (self.color, self.model, self.mpg)

    def drive_car(self):
        self.condition = "used"


class ElectricCar(Car):
    def __init__(self, battery_type):
        self.battery_type = battery_type

    def drive_car(self):
        self.condition = "like new"


my_car1 = Car("DeLorean", "silver", 88)
my_car = ElectricCar("molten salt")
my_car.model = "toyota"
my_car.color = 'brown'
my_car.mpg = 1000
print(my_car.condition)
my_car.drive_car()
print(my_car.condition)


class Point3D(object):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return ("(%d, %d, %d)" % (self.x, self.y, self.z))


my_point = Point3D(1, 2, 3)
print(my_point)
