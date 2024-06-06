import pygame

def play_audio(file_path):
    # Initialize the pygame mixer
    pygame.mixer.init()

    # Load a sound file
    sound_file = file_path
    sound = pygame.mixer.Sound(sound_file)

    # Set the volume (0.0 to 1.0, where 1.0 is the original volume)
    sound.set_volume(0.2)

    # Play the sound
    sound.play()

    # Wait for the sound to finish playing
    # pygame.time.wait(int(sound.get_length() * 1000))

    # 오디오 길이를 return
    return pygame.time.wait(int(sound.get_length()))
