import autopy as ap
ll
for i in range(330,400):
    copy = ap.bitmap.capture_screen()
    file = 'D:\Otepad\Screenshot '+str(i)+'.png'
    copy.save(file)