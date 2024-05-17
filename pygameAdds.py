import pygame

class buttons:
    pygame.font.init()
    myfont = pygame.font.Font("Formula1-Bold-4.ttf", 24)

    def __init__(self, coords, dimensions, label, colour=(0, 0, 0)):
        self.coords = coords
        self.dimensions = dimensions
        self.colour = colour
        self.hoverColour = self.colourCorrect(
            self.addListPar(self.colour, [20, 0, 20]))
        self.label = label

    def render(self, screen, mouse_pos):
        if self.check_hover(mouse_pos):
            pygame.draw.rect(screen, self.colour,
                             self.coords + self.dimensions)
        else:
            pygame.draw.rect(screen, self.hoverColour,
                             self.coords + self.dimensions)

        textsurface = self.myfont.render(self.label, False, (200, 0, 0))
        screen.blit(textsurface, self.addListPar(self.coords, (10, 12)))

    def check_hover(self, mouse_pos):
        if mouse_pos[0] > self.coords[0] and mouse_pos[0] < self.coords[
                0] + self.dimensions[0] and mouse_pos[1] > self.coords[
                    1] and mouse_pos[1] < self.coords[1] + self.dimensions[1]:
            return True
        return False

    def colourCorrect(self, colour):
        for i in colour:
            if i > 255:
                i = 255
            if i < 0:
                i = 0
        return colour

    def addListPar(self, l1, l2):
        if len(l1) != len(l2):
            return False
        newList = []
        for i in range(0, len(l1)):
            newList.append(l1[i]+l2[i])

        return newList

class inputField(buttons):
    def __init__(self, coords, dimensions, colour=(100, 100, 100)):
        buttons.__init__(self,coords, dimensions, "", colour)
        self.active = False
        self.charLim = 10

    def user_in(self, inp):
        try:
            if self.active:
                if inp == 8:
                    self.label=self.label[:-1]
                elif chr(inp).isalnum():
                    if len(self.label) <= self.charLim:
                        self.label=self.label+chr(inp)
        except:
            pass

    def userClicked(self):
        self.active = not self.active
