import retro
import numpy as np
import cv2
import neat
import pickle

env = retro.make('SonicTheHedgehog2-Genesis', 'EmeraldHillZone.Act1')

imgarray = []

xpos_end = 0

def eval_genomes(genomes, config):
    
    for genome_id, genome in genomes:
        ob = env.reset()
        ac = env.action_space.sample()

        inx, iny, inc = env.observation_space.shape

        inx = int(inx/8)
        iny = int(iny/8)

        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

        max_fitness = 0
        fitness_current = 0
        counter = 0
        xpos = 0
        rings = 0
        score = 0
        xpos_max = 0
        rings_max = 0
        score_max = 0        

        done = False
        # Voir l'image envoye au le neural network
        cv2.namedWindow('main', cv2.WINDOW_NORMAL)

        while not done:
	    #Voir l'image du jeu normale
            #env.render()
			
            # Voir l'image envoye au le neural network
            scaledimg = cv2.cvtColor(ob, cv2.COLOR_BGR2RGB)
            scaledimg = cv2.resize(scaledimg, (iny, inx))

            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx,iny))

            # Voir l'image envoye au le neural network
            cv2.imshow('main', scaledimg)
            cv2.waitKey(1)

            for x in ob:
                for y in x:
                    imgarray.append(y)

            nnOutput = net.activate(imgarray)

            ob, rew, done, info = env.step(nnOutput)
            imgarray.clear()

            xpos = info['x']
            rings = info['rings']
            score = info['score']
            xpos_end = info['screen_x_end']

            # Sonic fait du progress, fitness++
            if xpos > xpos_max:
                fitness_current += 2
                xpos_max = xpos

            # Score, fitness +
            if score > score_max:
                fitness_current += score*3
                score_max = score

            # + Rings, fitness++
            if rings > rings_max:
                fitness_current += (rings-rings_max)*2
                rings_max = rings

            # - Rings, fitness--
            if rings < rings_max:
                fitness_current -= (rings_max-rings)*2
                rings_max = rings

            # Sonic arrive a la fin du niveau
            if xpos == xpos_end:
                fitness_current += 5000
                print("WINNER")
                done = True

            # Compteur augmente si Sonic ne fait pas de progress
            if fitness_current > max_fitness:
                max_fitness = fitness_current
                counter = 0
            else:
                counter += 1

            if done or counter == 300:
                done = True
                print("Genome ", genome_id, "\tFitness ", fitness_current, "\tXpos ", xpos, "\tScore ", score, "\tRings ", rings)

            genome.fitness = fitness_current
            

#Fichier de configuration
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                    neat.DefaultSpeciesSet, neat.DefaultStagnation,
                    'config-feedforward')

#Creer une population
p = neat.Population(config)


#Statistiques avec NEAT
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10))


#Obtenir 
winner = p.run(eval_genomes)

with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)
