from brping import Ping1D
myPing = Ping1D()
myPing.connect_serial("com9", 9600)
#For UDP
#myPing.connect_udp("192.168.2.2", 9110)


data = myPing.get_distance()
if data:
    print("Distance: %s\tConfidence: %s%%" % (data["distance"], data["confidence"]))
else:
    print("Failed to get distance data")