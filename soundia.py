import pygame

pygame.init()
pygame.mixer.init()

pygame.mixer.music.load('VV.mp3')
pygame.mixer.music.play(-1)

clock = pygame.time.Clock()
while pygame.mixer.music.get_busy():
    clock.tick(1000)
    pygame.event.poll()
