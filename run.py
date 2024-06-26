import pygame
import os
import sys
from pygameAdds import buttons
# from test1 import run


swidth=1920
sheight=1514
sscale=0.5
scstartx=900
scstarty=550
scfacing=330
scscale=0.01
smap='./map.png'
sqfile='q_network_weightssilverstone.pth'
slidarval=23
sscheckpoints=[[(260, 116), (272, 142)],[(314, 102), (314, 132)],[(362, 104), (362, 134)],[(414, 104), (412, 136)],[(462, 112), (460, 140)],[(506, 118), (502, 144)],[(548, 156), (558, 126)],[(612, 138), (600, 168)],[(666, 150), (658, 180)],[(728, 164), (720, 194)],[(780, 176), (774, 210)],[(832, 174), (838, 204)],[(882, 170), (880, 200)],[(938, 194), (924, 226)],[(978, 222), (962, 250)],[(1406, 564),(1384, 588)],[(1030, 252), (1024, 280)],[(1080, 248), (1084, 274)],[(1144, 234), (1144, 266)],[(1180, 288), (1214, 278)],[(1206, 340), (1236, 332)],[(1222, 400), (1256, 396)],[(1254, 450), (1278, 434)],[(1292, 490), (1318, 476)],[(1328, 530), (1352, 510)],[(1446, 650), (1472, 630)],[(1490, 696), (1514, 674)],[(1538, 748), (1564, 730)],[(1582, 794), (1608, 772)],[(1620, 828), (1646, 814)],[(1662, 874), (1688, 856)],[(1712, 934), (1746, 928)],[(1750, 984), (1786, 984)],[(1780, 1070), (1810, 1076)],[(1714, 1134), (1720, 1166)],[(1642, 1148), (1648, 1178)],[(1574, 1172), (1580, 1200)],[(1510, 1196), (1520, 1220)],[(1452, 1220), (1460, 1252)],[(1368, 1332), (1402, 1336)],[(1276, 1380), (1274, 1412)],[(1160, 1372), (1186, 1352)],[(1132, 1324), (1156, 1304)],[(1104, 1280), (1128, 1262)],[(1102, 1226), (1074, 1236)],[(1044, 1198), (1074, 1184)],[(1044, 1146), (1012, 1156)],[(992, 1124), (1016, 1106)],[(990, 1070), (962, 1082)],[(944, 1054), (966, 1034)],[(934, 982), (904, 992)],[(882, 944), (914, 940)],[(914, 902), (886, 894)],[(910, 838), (940, 848)],[(960, 800), (934, 788)],[(946, 726), (978, 726)],[(966, 668), (934, 668)],[(920, 614), (948, 612)],[(910, 508), (878, 508)],[(1042, 422), (1012, 432)],[(990, 434), (1000, 468)],[(944, 436), (956, 472)],[(940, 382), (952, 350)],[(860, 374), (840, 352)],[(762, 394), (780, 422)],[(708, 424), (722, 456)],[(666, 450), (682, 480)],[(612, 484), (624, 510)],[(566, 508), (578, 534)],[(490, 548), (510, 578)],[(432, 588), (450, 616)],[(384, 610), (398, 642)],[(320, 654), (346, 674)],[(268, 700), (302, 706)],[(260, 738), (296, 738)],[(302, 782), (320, 760)],[(360, 820), (392, 802)],[(384, 858), (412, 864)],[(360, 888), (368, 926)],[(326, 902), (326, 932)],[(274, 908), (292, 878)],[(254, 846), (224, 856)],[(192, 826), (214, 808)],[(180, 772), (148, 780)],[(122, 736), (154, 728)],[(106, 660), (136, 660)],[(118, 578), (144, 580)],[(126, 514), (158, 514)],[(138, 460), (168, 460)], [(150, 406), (180, 410)],[(158, 352), (190, 356)],[(170, 304), (200, 308)],[(200, 196), (228, 210)], [(216, 160), (242, 182)],[(190, 240), (216, 250)], [(930, 566), (896, 568)]]

#nascar
nwidth=2200
nheight=1514
nscale=0.6
ncstartx=502
ncstarty=370
ncfacing=90
ncscale=0.02
nmap='./map2.png'
nqfile='q_network_weightsnascar.pth'
nlidarval=60
nascarcheckpoints = [[[408, 372], [430, 424]], [[801, 331], [808, 394]], [[927, 335], [936, 396]], [[1074, 332], [1078, 394]], [[1221, 337], [1225, 389]], [[1354, 335], [1360, 392]], [[1503, 334], [1498, 388]], [[1637, 332], [1632, 395]], [[1751, 355], [1734, 408]], [[1843, 400], [1821, 455]], [[1922, 464], [1883, 510]], [[1980, 544], [1934, 584]], [[2017, 636], [1964, 661]], [[2031, 742], [1977, 749]], [[2027, 834], [1970, 834]], [[1997, 928], [1948, 901]], [[1938, 1028], [1887, 1003]], [[1850, 1101], [1807, 1069]], [[1747, 1156], [1716, 1105]], [[1665, 1174], [1641, 1114]], [[1578, 1174], [1556, 1112]], [[1460, 1174], [1452, 1117]], [[1343, 1175], [1327, 1117]], [[1207, 1115], [1218, 1178]], [[1114, 1116], [1120, 1172]], [[1029, 1114], [1032, 1172]], [[956, 1115], [957, 1176]], [[844, 1117], [835, 1182]], [[724, 1114], [722, 1182]], [[620, 1116], [621, 1172]], [[524, 1111], [485, 1165]], [[423, 1080], [375, 1123]], [[337, 1022], [300, 1068]], [[283, 958], [232, 998]], [[245, 887], [181, 894]], [[227, 805], [161, 808]], [[227, 728], [167, 720]], [[241, 636], [184, 608]], [[267, 571], [222, 530]], [[340, 489], [298, 436]], [[704, 335], [707, 392]],[[570, 334], [575, 394]]]


def main():
    windowX = 800
    windowY = 600

    pygame.init()
    screen = pygame.display.set_mode((windowX, windowY),pygame.RESIZABLE)
    pygame.mouse.set_cursor(*pygame.cursors.diamond)
    logo=pygame.image.load('./f1logo.png').convert_alpha()
    logo = pygame.transform.smoothscale(logo, (100, 100))

    onScreenButtons=[]

    bg = pygame.image.load(os.path.dirname(__file__) + './bg.png')
    pygame.display.set_caption("F1")

    names = ["Silverstone", "Nascar", "Silverstone - A*","Nascar - A*"]
    for i in range(0,4):
        but = buttons((285, 20 + 60 * (i + 1)), (230, 50), names[i])
        onScreenButtons.append(but)

    running = True
    while running:  # Main Screen Loop
        screen.fill((0, 0, 0))  # R G B
        screen.blit(bg, (0, 0))
        screen.blit(logo,(30,0))
        # Render each button
        for i in onScreenButtons:
            i.render(screen, pygame.mouse.get_pos())

        # Controls
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                for i in onScreenButtons:
                    if i.check_hover(pygame.mouse.get_pos()):
                        if(i.label == names[0]):
                            os.system("python test1.py 2 &")
                            exit()
                        elif(i.label == names[1]):
                            os.system("python test1.py 1 &")
                            exit()
                        if(i.label == names[2]):
                            os.system("python test1.py 4 &")
                            exit()
                        elif(i.label == names[3]):
                            os.system("python test1.py 3 &")
                            exit()

        pygame.display.update()

    pygame.quit()

if __name__ == "__main__":
    main()
