import pygame
import os
import time
from gameFiles.pygameAdds import buttons, inputField
from gameFiles.database import database

def main(screen, games):
    gameDatabase = database(disable=False)
    onScreenButtons=[]

    bg = pygame.image.load(os.path.dirname(__file__) + '/bg.png')
    pygame.display.set_caption("F1")

    for i in range(len(games)):
        but = buttons((300, 150 + 60 * (i + 1)), (200, 50), games[i])
        onScreenButtons.append(but)
    inputBox = inputField((500, 10), (300, 40))

    inputBox.label=gameDatabase.getCurrentUser()

    while True:  # Main Screen Loop
        screen.fill((0, 0, 0))  # R G B
        screen.blit(bg, (0, 0))
        pygame.time.delay(10)

        # Render each button for eeach frame, also inputField
        for i in onScreenButtons:
            i.render(screen, pygame.mouse.get_pos())
        inputBox.render(screen, pygame.mouse.get_pos())

        # Controls
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True, None

            elif event.type == pygame.KEYDOWN:
                if event.key == 1073742049:
                    return True, None
                inputBox.user_in(event.key)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if inputBox.check_hover(pygame.mouse.get_pos()):
                    inputBox.userClicked()

                if not gameDatabase.isRegistered(inputBox.label):
                    break

                for i in onScreenButtons:
                    if i.check_hover(pygame.mouse.get_pos()):
                        gameDatabase.setCurrentUser(inputBox.label)
                        return False, i.label

        pygame.display.update()
