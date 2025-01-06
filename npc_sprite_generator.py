import random
from os.path import exists
import os
import numpy as np
import math
from collections import defaultdict
import PIL
from PIL import Image, ImageOps, ImageFilter
from matplotlib import cm
import sys
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from scipy.ndimage import rotate


 


CRYPTIC_OCEAN_PAL = ["2a173b", "3f2c5f", "443f7b", "4c5c87", "69809e", "95c5ac"]
GALACTIC_GREENS_PAL = ["e8f1a6", "cad670", "a3c255", "6fa341", "498f45", "387450", "2d5c56", "1f3741", "1e2029", "16161c"]

BORKFEST_PAL = ["dfd785", "ebc275", "f39949", "ff7831", "ca5a2e", "963c3c", "3a2802", "202215"]
VIVID_SCREAM_PAL = ["561d17", "5c4fa3", "74af34", "caf532"]
RUNE_SPELL_PAL = ["000412", "d9b982", "f6cd26"]
SMOKY_PAL = ["fafafa", "d4d8e0", "acacac", "918b8c", "6b615e", "3b342e", "24211a", "0e0d0a", "030201"]
GOLD_PAL = ["f6cd26", "ac6b26", "563226", "331c17", "bb7f57", "725956", "393939", "202020"]

rand1 = ["7ac74f","a1cf6b","d5d887","e0c879","e87461"]
rand2 = ["5b5b5b","7d7c7a","c9c19f","edf7d2","edf7b5"]
rand3 = ["e5ffde","bbcbcb","9590a8","634b66","18020c"]
rand4 = ["141204","262a10","54442b","a9714b","e8985e"]
rand5 = ["b37ba4","d99ac5","dccde8","14bdeb","00100b"]


pal_list = [BORKFEST_PAL, VIVID_SCREAM_PAL, RUNE_SPELL_PAL, SMOKY_PAL, GOLD_PAL, CRYPTIC_OCEAN_PAL, GALACTIC_GREENS_PAL, rand1, rand2, rand3, rand4, rand5]






def hex_to_rgba(value):
	"""Return (red, green, blue) for the color given as #rrggbb."""
	value = value.lstrip('#')
	lv = len(value)
	return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3)) + (255,)

def round_to_closest_even_number(number):
	return 2 * int(number / 2)



class ColourPalGen(object):

	def __init__(self, max_r=5, max_g=5, max_b=5):
		self.max_r = max_r
		self.max_g = max_g
		self.max_b = max_b


	def getPal(self, num_colours):

		if num_colours > self.max_r*self.max_g*self.max_b:
			print("Can't return a pal for that amount of colours with the max_r, max_g and max_b contraits given. Please change those values.")
			exit()

		random_baseline = random.randint(0, 255)

		r_mul = random.randint(1, self.max_r)
		g_mul = random.randint(1, self.max_g)
		b_mul = random.randint(1, self.max_b)

		r_increments = int(256/r_mul)
		g_increments = int(256/g_mul)
		b_increments = int(256/b_mul)

		r_vals = []
		r_base = random_baseline
		for mul in range(1, r_mul + 1):
			r_base = (r_base + r_increments)%255
			r_vals.append(r_base)

		g_vals = []
		g_base = random_baseline
		for mul in range(1, g_mul + 1):
			g_base = (g_base + g_increments)%255
			g_vals.append(g_base)

		b_vals = []
		b_base = random_baseline
		for mul in range(1, b_mul + 1):
			b_base = (b_base + b_increments)%255
			b_vals.append(b_base)


		total_colour_pal = []

		for r in r_vals:
			for g in g_vals:
				for b in b_vals:
					total_colour_pal.append((r,g,b))

		colour_pal = random.choices(total_colour_pal, k=num_colours)

		return colour_pal












class BottomUpGenerator(object):

	floor_char = '.'
	empty_char = '_'
	wall_char = 'x'
	detail_one_char = 'm'
	detail_two_char = 'r'
	detail_three_char = 'f'

	empty_will_be = False

	num_tries_before_change_par = 15

	def __init__(self, canvas_size_x, canvas_size_y, dungeon_density, num_loops, neighbour_depth, neighbour_number_threshold, detail_one_density, detail_two_density, detail_three_density, symmetry = "none", mutation_rate = 0.0):

		self.map = []
		self.canvas_size_x = canvas_size_x
		self.canvas_size_y = canvas_size_y
		self.dungeon_density = dungeon_density
		self.num_loops = num_loops
		self.neighbour_depth = neighbour_depth
		self.neighbour_number_threshold = neighbour_number_threshold
		self.detail_one_density = detail_one_density
		self.detail_two_density = detail_two_density
		self.detail_three_density = detail_three_density
		self.small_col_matrix = None
		self.symmetry = symmetry
		self.mutation_rate = mutation_rate


	def network_from_matrix(self, matrix):

		G = nx.Graph()

		for i in range(self.canvas_size_y):
			for j in range(self.canvas_size_x):
				if matrix[i][j] != 1:
					if i > 0:
						if matrix[i-1][j]:
							G.add_edge((i-1,j), (i,j))
					if j > 0:
						if matrix[i][j-1] != 1:
							G.add_edge((i,j-1), (i,j))
					if i < len(matrix) - 1:
						if matrix[i+1][j] != 1:
							G.add_edge((i+1,j), (i,j))
					if j < len(matrix[0]) - 1:
						if matrix[i][j+1] != 1:
							G.add_edge((i,j+1), (i,j))
		return G


	def check_if_connected(self, array):
		G = self.network_from_matrix(array)
		return nx.is_connected(G)

	def mutateKeepShape(self, array, mutation_rate):

		new_array = array.copy()

		for i in range(len(array)):
			for j in range(len(array[0])):
				if array[i][j] == 1:
					pass
				elif array[i][j] == 0.7:
					array[i][j] == 0.5
				else:
					if random.uniform(0,100) < mutation_rate:
						searching = True
						while searching:
							mut_i = random.randint(0, len(array) - 1)
							mut_j = random.randint(0, len(array[0]) - 1)

							if array[mut_i][mut_j] == 1:
								pass
							else:
								new_array[i][j] = array[mut_i][mut_j]
								searching = False
		return new_array


	def side_mutate(self):


		self.side_matrix = np.array(self.pre_wall_map)

		num_loops = 2





	def make_a_matrix(self, verbose = False, name = None):

		# print(self.canvas_size_x)
		# print(self.canvas_size_x)

		if self.canvas_size_x < 3 or self.canvas_size_y < 3:
			self.empty_will_be = True
			return

		connected = False

		counter = 0

		while( connected == False ):

			if counter > self.num_tries_before_change_par:
				self.dungeon_density += 1

			self.map = []

			self.initialize_empty_map()

			if verbose:
				self.print_map()

			self.randomly_add_floor_tiles()

			if verbose:
				self.print_map()

			self.aglutinate_floor()

			self.pre_wall_map = self.map.copy()

			self.wall_it_up()

			if verbose:
				self.print_map()

			np.set_printoptions(threshold=np.inf)

			self.place_details_one_randomly()

			if verbose:
				self.print_map()

			self.place_details_two_randomly()

			if verbose:
				self.print_map()

			self.place_details_three_randomly()

			if verbose:
				self.print_map()

			self.front_array = np.random.rand(self.canvas_size_y, self.canvas_size_x)

			for line in range(self.canvas_size_y):
				for column in range(self.canvas_size_x):
					if self.map[line][column] == self.floor_char:
						self.front_array[line][column] = 0
					elif self.map[line][column] == self.wall_char:
						self.front_array[line][column] = 0.5
					elif self.map[line][column] == self.empty_char:
						self.front_array[line][column] = 1
					else:
						self.front_array[line][column] = 0.7

			self.original_array = self.front_array

			try:
				connected = self.check_if_connected(self.front_array)
				if connected == False:
					continue
			except nx.exception.NetworkXPointlessConcept:
				connected = False
				continue

			if self.symmetry == "none":
				pass

			elif self.symmetry == "vertical":

				self.front_array = self.front_array[:,0:int(self.canvas_size_x/2)]

				flip_front_array = np.flip(self.front_array, 1)

				self.front_array = np.append(self.front_array, flip_front_array, axis = 1)

			elif self.symmetry == "horizontal":

				self.front_array = self.front_array[0:int(self.canvas_size_y/2)]

				flip_front_array = np.flip(self.front_array, 0)

				self.front_array = np.append(self.front_array, flip_front_array, axis = 0)

			else:
				print("Invalid Value for Symmetry!")
				exit()

			counter += 1

			try:
				connected = self.check_if_connected(self.front_array)
			except nx.exception.NetworkXPointlessConcept:
				connected = False
	

			self.back_array = self.mutateKeepShape(self.front_array, self.mutation_rate)

			if self.symmetry == "none":
					pass

			elif self.symmetry == "vertical":

				self.back_array = self.back_array[:,0:int(self.canvas_size_x/2)]

				flip_back_array = np.flip(self.back_array, 1)

				self.back_array = np.append(self.back_array, flip_back_array, axis = 1)

			elif self.symmetry == "horizontal":

				self.back_array = self.back_array[0:int(self.canvas_size_y/2)]

				flip_back_array = np.flip(self.back_array, 0)

				self.back_array = np.append(self.back_array, flip_back_array, axis = 0)

			else:
				print("Invalid Value for Symmetry!")
				exit()

			try:
				connected = self.check_if_connected(self.back_array)
			except nx.exception.NetworkXPointlessConcept:
				connected = False




			self.hair_array = self.mutateKeepShape(self.original_array, self.mutation_rate)

			if self.symmetry == "none":
					pass

			elif self.symmetry == "vertical":

				self.hair_array = self.hair_array[:,int(self.canvas_size_x/2):self.canvas_size_x]

				flip_hair_array = np.flip(self.hair_array, 1)

				self.hair_array = np.append(flip_hair_array, self.hair_array,  axis = 1)

			elif self.symmetry == "horizontal":

				self.hair_array = self.hair_array[int(self.canvas_size_y/2):]

				flip_hair_array = np.flip(self.hair_array, 0)

				self.hair_array = np.append(flip_hair_array, self.hair_array, axis = 0)

			else:
				print("Invalid Value for Symmetry!")
				exit()

			try:
				connected = self.check_if_connected(self.hair_array)
			except nx.exception.NetworkXPointlessConcept:
				connected = False



		self.side_mutate()

		#self.wall_it_up(map_n = self.side_matrix)


		self.side_array = np.random.rand(self.canvas_size_y, self.canvas_size_x)

		for line in range(self.canvas_size_y):
			for column in range(self.canvas_size_x):
				if self.side_matrix[line][column] == self.floor_char:
					self.side_array[line][column] = 0
				elif self.side_matrix[line][column] == self.wall_char:
					self.side_array[line][column] = 0.5
				elif self.side_matrix[line][column] == self.empty_char:
					self.side_array[line][column] = 1
				else:
					self.side_array[line][column] = 0.7

			




	def print_map(self):

		print("--------MAP--------")

		for line in self.map:
			for char in line:
				print(char, end = '')
			print()



	def initialize_empty_map(self):

		for _ in range(self.canvas_size_y):
			line = ['_']*self.canvas_size_x
			self.map.append(line)




	def randomly_add_floor_tiles(self):
		for i in range(self.canvas_size_y):
			for j in range(self.canvas_size_x):
				if random.randint(0,100) < self.dungeon_density:
					self.map[i][j] = self.floor_char

	def get_position_empty_neighbours(self, i, j):

		neighbours = []

		if self.neighbour_depth <= 0:
			print("Depth variable needs to be greater than 0")
			exit()

		for n_i in range(i - 1, i + 1 + 1):
			for n_j in range(j - 1, j + 1 + 1):
				if n_i == i and n_j == j:
					continue
				if self.map[n_i][n_j] == self.empty_char:
					neighbours.append([n_i, n_j])

		return neighbours

	def get_position_floored_neighbours(self, i, j, map_n = None):

		neighbours = []
		bordering_edges = False

		if map_n is None:
			map_n = self.map

		if self.neighbour_depth <= 0:
			print("Depth variable needs to be greater than 0")
			exit()

		for n_i in range(i - self.neighbour_depth, i + self.neighbour_depth + 1):
			for n_j in range(j - self.neighbour_depth, j + self.neighbour_depth + 1):


				if n_i < 0 or n_i >= self.canvas_size_y or n_j < 0 or n_j >= self.canvas_size_x:
					bordering_edges = True
					continue
				if n_i == i and n_j == j:
					continue
				if map_n[n_i][n_j] == self.floor_char:
					neighbours.append([n_i, n_j])

		return neighbours, bordering_edges


	def aglutinate_floor(self, map_n = None):

		if map_n is None:
			map_n = self.map

		for run in range(self.num_loops):
			to_add = []
			to_remove = []
			for i in range(self.canvas_size_y):
				for j in range(self.canvas_size_x):
					neighbors, bordering = self.get_position_floored_neighbours(i, j)
					if (len(neighbors) >= self.neighbour_number_threshold) and not bordering:
						to_add.append([i, j])
					else:
						to_remove.append([i,j])
			for cord in to_add:
				map_n[cord[0]][cord[1]] = self.floor_char
			for cord in to_remove:
				map_n[cord[0]][cord[1]] = self.empty_char




	def wall_it_up(self, map_n = None):

		if map_n is None:
			map_n = self.map


		for i in range(self.canvas_size_y):
			for j in range(self.canvas_size_x):
				if map_n[i][j] == self.floor_char:
					empty_neighbours = self.get_position_empty_neighbours(i,j)
					for neighbour in empty_neighbours:
						map_n[neighbour[0]][neighbour[1]] = self.wall_char



	def place_details_one_randomly(self):

		for _ in range(self.canvas_size_x*self.canvas_size_y):
			if self.canvas_size_y-2 > 1 and self.canvas_size_x-2 > 1:
				i = random.randint(1, self.canvas_size_y-2)
				j = random.randint(1, self.canvas_size_x-2)
				if self.map[i][j] == self.floor_char and len(self.get_position_floored_neighbours(i,j)[0]) == 8:
					if random.uniform(0.0, 100.0) < self.detail_one_density:
						self.map[i][j] = self.detail_one_char


	def place_details_two_randomly(self):

		for _ in range(self.canvas_size_x*self.canvas_size_y):
			if self.canvas_size_y-2 > 1 and self.canvas_size_x-2 > 1:
				i = random.randint(1, self.canvas_size_y-2)
				j = random.randint(1, self.canvas_size_x-2)
				if self.map[i][j] == self.floor_char and len(self.get_position_floored_neighbours(i,j)[0]) == 8:
					if random.uniform(0.0, 100.0) < self.detail_two_density:
						self.map[i][j] = self.detail_two_char

	def place_details_three_randomly(self):

		for _ in range(self.canvas_size_x*self.canvas_size_y):
			if self.canvas_size_y-2 > 1 and self.canvas_size_x-2 > 1:
				i = random.randint(1, self.canvas_size_y-2)
				j = random.randint(1, self.canvas_size_x-2)
				if self.map[i][j] == self.floor_char and len(self.get_position_floored_neighbours(i,j)[0]) == 8:
					if random.uniform(0.0, 100.0) < self.detail_three_density:
						self.map[i][j] = self.detail_three_char



	def find_coeffs(self, pa, pb):
		matrix = []
		for p1, p2 in zip(pa, pb):
			matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
			matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

		A = np.matrix(matrix, dtype=np.float)
		B = np.array(pb).reshape(8)

		res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
		return np.array(res).reshape(8)


	def simple_triplicate(self, im):

		width, height = im.size
		m = 0.05
		xshift = 0
		new_width = width + int(round(xshift))
		tim = im.transform((width*3, height), Image.AFFINE,
				(1, m, -xshift if m > 0 else 0, 0, 1, 0), Image.BICUBIC)
		

		width, height = im.size
		m = -0.05
		new_width = width + int(round(xshift))
		kim = im.transform((width, height), Image.AFFINE,
				(1, m, 0, 0, 1, 0), Image.BICUBIC)
		kim.paste(tim, (self.canvas_size_x,0))
		#kim.save("right_1.png","PNG")


		tim.paste(im, (self.canvas_size_x,0))
		tim.paste(kim, (self.canvas_size_x*2,0))

		return tim


	def simple_blur(self, im):
		coeffs = self.find_coeffs(
		[(0, 0), (self.canvas_size_x, 0), (self.canvas_size_x, self.canvas_size_y), (0, self.canvas_size_y)],
		[(-2, -2), (self.canvas_size_x + 2, -2), (self.canvas_size_x + 2, self.canvas_size_y + 2), (-2, self.canvas_size_y + 2)])

		width, height = im.size
		im = im.transform((width, height), Image.AFFINE,
				 coeffs, Image.BICUBIC)
		return im


	def get_image(self, pal):


		#Making front image

		colours = [random.choice(pal),random.choice(pal),random.choice(pal),random.choice(pal)]
		

		front_im_list = []

		back_im_list = []

		side_im_list = []

		hair_im_list = []

		if self.empty_will_be:
			front_im = Image.new('RGBA', (10, 10))
			front_im_list.append(front_im)

			back_im = Image.new('RGBA', (10, 10))
			back_im_list.append(back_im)

			side_im = Image.new('RGBA', (10, 10))
			side_im_list.append(side_im)

			hair_im = Image.new('RGBA', (10, 10))
			hair_im_list.append(hair_im)

			return front_im_list, back_im_list, side_im_list, hair_im_list

		front_im = Image.fromarray(np.uint8(cm.gist_earth(self.front_array)*255))

		front_im = front_im.convert("RGBA")

		datas = front_im.getdata()

		newData = []

		for item in datas:
			#print(item)
			if item[0] == 253 and item[1] == 250 and item[2] == 250:
				newData.append((255, 255, 255, 0)) #This is invisible
			elif item[0] == 93 and item[1] == 160 and item[2] == 75:
				newData.append(colours[1])
			elif item[0] == 183 and item[1] == 181 and item[2] == 94:
				newData.append(colours[0])
			else:
				newData.append(colours[2])

		front_im.putdata(newData)

		front_im_list.append(front_im)



		back_im = Image.fromarray(np.uint8(cm.gist_earth(self.back_array)*255))

		back_im = back_im.convert("RGBA")

		datas = back_im.getdata()

		newData = []

		for item in datas:
			#print(item)
			if item[0] == 253 and item[1] == 250 and item[2] == 250:
				newData.append((255, 255, 255, 0)) #This is invisible
			elif item[0] == 93 and item[1] == 160 and item[2] == 75:
				newData.append(colours[1])
			elif item[0] == 183 and item[1] == 181 and item[2] == 94:
				newData.append(colours[1])
			else:
				newData.append(colours[2])

		back_im.putdata(newData)

		back_im_list.append(back_im)



		side_im = Image.fromarray(np.uint8(cm.gist_earth(self.side_array)*255))

		side_im = side_im.convert("RGBA")

		datas = side_im.getdata()

		newData = []

		for item in datas:
			#print(item)
			if item[0] == 253 and item[1] == 250 and item[2] == 250:
				newData.append((255, 255, 255, 0)) #This is invisible
			elif item[0] == 93 and item[1] == 160 and item[2] == 75:
				newData.append(colours[1])
			elif item[0] == 183 and item[1] == 181 and item[2] == 94:
				newData.append(colours[0])
			else:
				newData.append(colours[2])

		side_im.putdata(newData)

		side_im_list.append(side_im)


		hair_im = Image.fromarray(np.uint8(cm.gist_earth(self.hair_array)*255))

		hair_im = hair_im.convert("RGBA")

		datas = hair_im.getdata()

		newData = []

		for item in datas:
			#print(item)
			if item[0] == 253 and item[1] == 250 and item[2] == 250:
				newData.append((255, 255, 255, 0)) #This is invisible
			elif item[0] == 93 and item[1] == 160 and item[2] == 75:
				newData.append(colours[1])
			elif item[0] == 183 and item[1] == 181 and item[2] == 94:
				newData.append(colours[1])
			else:
				newData.append(colours[2])

		hair_im.putdata(newData)

		hair_im_list.append(hair_im)

		return front_im_list, back_im_list, side_im_list, hair_im_list

		

class HeadGenerator(object):

	def getHead(self, size_x, size_y, pal):

		# canvas_size_x, canvas_size_y, dungeon_density, num_loops, neighbour_depth, neighbour_number_threshold, detail_one_density, detail_two_density, detail_three_density, mutation_rate
		bug = BottomUpGenerator(size_x, size_y, 20, 3, 1, 3, 10, 10, 10, symmetry = "vertical", mutation_rate = 10)

		bug.make_a_matrix()

		im_list, bim_list, sim_list, him_list = bug.get_image(pal)

		return im_list, bim_list, sim_list, him_list


class TorsoGenerator(object):

	def getTorso(self, size_x, size_y, pal):

		# canvas_size_x, canvas_size_y, dungeon_density, num_loops, neighbour_depth, neighbour_number_threshold, detail_one_density, detail_two_density, detail_three_density, mutation_rate
		bug = BottomUpGenerator(size_x, size_y, 20, 3, 1, 3, 0.5, 0.5, 0.4, symmetry = "vertical", mutation_rate = 10)

		bug.make_a_matrix()

		im_list, bim_list, sim_list, him_list = bug.get_image(pal)

		return im_list, bim_list, sim_list, him_list


class LimbGenerator(object):

	def getLimbs(self, size_x, size_y, pal):

		# canvas_size_x, canvas_size_y, dungeon_density, num_loops, neighbour_depth, neighbour_number_threshold, detail_one_density, detail_two_density, detail_three_density, mutation_rate
		bug = BottomUpGenerator(size_x, size_y, 20, 3, 1, 3, 0.5, 0.5, 0.4, symmetry = "none", mutation_rate = 10)

		bug.make_a_matrix()

		left_limb_list, back_left_limb_list, side_left_limb_list, _ = bug.get_image(pal)

		right_limb_list = []
		back_right_limb_list = []
		side_right_limb_list = []

		for i in range(len(left_limb_list)):

			right_limb = left_limb_list[i].transpose(Image.FLIP_LEFT_RIGHT)
			back_right_limb = back_left_limb_list[i].transpose(Image.FLIP_LEFT_RIGHT)
			side_right_limb = side_left_limb_list[i].transpose(Image.FLIP_LEFT_RIGHT)

			right_limb_list.append(right_limb)
			back_right_limb_list.append(back_right_limb)
			side_right_limb_list.append(side_right_limb)


		return left_limb_list, back_left_limb_list, side_left_limb_list, right_limb_list, back_right_limb_list, side_right_limb_list


class NPC_Limb_Movement_Generator(object):

	def __init__(self):

		img = plt.imread("/Users/stuff/Desktop/Infinite-Game/player.png")

		fig, ax = plt.subplots(1)
		#ax.set_aspect('equal')

		ax.imshow(img)

		# Create the base ellipse
		ellipse = Ellipse((30, 30), width=40, height=10,
		                  edgecolor='black', facecolor='none', linewidth=2)

		ax.add_patch(ellipse)

		# Get the path
		path = ellipse.get_path()
		# Get the list of path vertices
		vertices = path.vertices.copy()
		# Transform the vertices so that they have the correct coordinates
		vertices = ellipse.get_patch_transform().transform(vertices)

		print(vertices)

		# You can then save the vertices array to a file: csv, pickle... It's up to you

		plt.show()



class NPC_Humanoid_Generator(object):

	def __init__(self, width, height, size_multiplication_list):

		self.width = width

		self.height = height

		self.size_multiplication_list = size_multiplication_list

		self.torso_gen = TorsoGenerator()

		self.limb_gen = LimbGenerator()

		self.head_gen = HeadGenerator()

		self.sprite_sheet_list = []



	def makeNPC(self, torso_dim = (0.6, 0.60), torso_offset = (0.0, 0.0), arm_dim = (0.25, 0.40), arm_offset = (0.2, 0.0), leg_dim = (0.25, 0.4), leg_offset = (0.0, 0.0), head_dim = (0.5, 0.5), head_offset = (0.0, 0.0), head_wobble = (0,2), torso_wobble = (0,1), arm_wobble = (1,1), side_arm_wobble = (2,0), leg_wobble = (0,1), side_leg_wobble = (2,0), save_name = None, side_arm_angles = (0, -25, 25), c_palette = None):

		global pal_list

		
	
		if c_palette is None:
			colly = ColourPalGen()
			pal = colly.getPal(5)
		else:
			pal = c_palette

		#Making Torso
		torso_list, back_torso_list, side_torso_list, hair_torso_list = self.torso_gen.getTorso(round_to_closest_even_number(self.width * torso_dim[0]), round_to_closest_even_number(self.height * torso_dim[1]), pal)
		left_arm_list, back_left_arm_list, side_left_arm_list, right_arm_list, back_right_arm_list, side_right_arm_list = self.limb_gen.getLimbs(round_to_closest_even_number(self.width * arm_dim[0]), round_to_closest_even_number(self.height * arm_dim[1]), pal)
		left_leg_list, back_left_leg_list, side_left_leg_list, right_leg_list, back_right_leg_list, side_right_leg_list = self.limb_gen.getLimbs(round_to_closest_even_number(self.height * leg_dim[0]), round_to_closest_even_number(self.width * leg_dim[1]), pal)
		head_list, back_head_list, side_head_list, hair_head_list = self.head_gen.getHead(round_to_closest_even_number(self.width * head_dim[0]), round_to_closest_even_number(self.width * head_dim[1]), pal)

		output_size = 0



		torso = torso_list[output_size]
		#torso.resize((torso.width*20, torso.height*20),resample=Image.NEAREST).save("paper_images/"+  "torso"  + ".png","PNG")
		back_torso = back_torso_list[output_size]
		#back_torso.resize((back_torso.width*20, back_torso.height*20),resample=Image.NEAREST).save("paper_images/"+  "back_torso"  + ".png","PNG")
		side_torso = side_torso_list[output_size]
		#side_torso.resize((side_torso.width*20, side_torso.height*20),resample=Image.NEAREST).save("paper_images/"+  "side_torso"  + ".png","PNG")
		hair_torso = hair_torso_list[output_size]
		#hair_torso.resize((hair_torso.width*20, hair_torso.height*20),resample=Image.NEAREST).save("paper_images/"+  "hair_torso"  + ".png","PNG")

		left_arm= left_arm_list[output_size]
		#left_arm.resize((left_arm.width*20, left_arm.height*20),resample=Image.NEAREST).save("paper_images/"+  "left_arm"  + ".png","PNG")
		back_left_arm = back_left_arm_list[output_size]
		#back_left_arm.resize((back_left_arm.width*20, back_left_arm.height*20),resample=Image.NEAREST).save("paper_images/"+  "back_left_arm"  + ".png","PNG")
		side_left_arm = side_left_arm_list[output_size]
		#side_left_arm.resize((side_left_arm.width*20, side_left_arm.height*20),resample=Image.NEAREST).save("paper_images/"+  "side_left_arm"  + ".png","PNG")
		right_arm = right_arm_list[output_size]
		#right_arm.resize((right_arm.width*20, right_arm.height*20),resample=Image.NEAREST).save("paper_images/"+  "right_arm"  + ".png","PNG")
		back_right_arm = back_right_arm_list[output_size]
		#back_right_arm.resize((back_right_arm.width*20, back_right_arm.height*20),resample=Image.NEAREST).save("paper_images/"+  "back_right_arm"  + ".png","PNG")
		side_right_arm = side_right_arm_list[output_size]
		#side_right_arm.resize((side_right_arm.width*20, side_right_arm.height*20),resample=Image.NEAREST).save("paper_images/"+  "side_right_arm"  + ".png","PNG")

		left_leg = left_leg_list[output_size]
		#left_leg.resize((left_leg.width*20, left_leg.height*20),resample=Image.NEAREST).save("paper_images/"+  "left_leg"  + ".png","PNG")
		back_left_leg = back_left_leg_list[output_size]
		#back_left_leg.resize((back_left_leg.width*20, back_left_leg.height*20),resample=Image.NEAREST).save("paper_images/"+  "back_left_leg"  + ".png","PNG")
		side_left_leg = side_left_leg_list[output_size]
		#side_left_leg.resize((side_left_leg.width*20, side_left_leg.height*20),resample=Image.NEAREST).save("paper_images/"+  "side_left_leg"  + ".png","PNG")
		right_leg = right_leg_list[output_size]
		#right_leg.resize((right_leg.width*20, right_leg.height*20),resample=Image.NEAREST).save("paper_images/"+  "right_leg"  + ".png","PNG")
		back_right_leg = back_right_leg_list[output_size]
		#back_right_leg.resize((back_right_leg.width*20, back_right_leg.height*20),resample=Image.NEAREST).save("paper_images/"+  "back_right_leg"  + ".png","PNG")
		side_right_leg = side_right_leg_list[output_size]
		#side_right_leg.resize((side_right_leg.width*20, side_right_leg.height*20),resample=Image.NEAREST).save("paper_images/"+  "side_right_leg"  + ".png","PNG")

		head = head_list[output_size]
		#head.resize((head.width*20, head.height*20),resample=Image.NEAREST).save("paper_images/"+  "head"  + ".png","PNG")
		back_head = back_head_list[output_size]
		#back_head.resize((back_head.width*20, back_head.height*20),resample=Image.NEAREST).save("paper_images/"+  "back_head"  + ".png","PNG")
		side_head = side_head_list[output_size]
		#side_head.resize((side_head.width*20, side_head.height*20),resample=Image.NEAREST).save("paper_images/"+  "side_head"  + ".png","PNG")
		hair_head = hair_head_list[output_size]
		#hair_head.resize((hair_head.width*20, hair_head.height*20),resample=Image.NEAREST).save("paper_images/"+  "hair_head"  + ".png","PNG")



		############################################################
		#														   #
		#				Making Front Facing Sprites 			   #
		#														   #
		############################################################

		front_triad = Image.new("RGBA", ((self.width)*3, (self.height)), (0,0,0,0))


		# Making middle front facing
		middle_front = Image.new("RGBA", ((self.width), (self.height)), (0,0,0,0))

		middle_front.paste(hair_head, (int(((self.width) - hair_head.size[0])/2) + int(head_offset[0]*self.width), 0 + head_wobble[1] + int(head_offset[1]*self.height)), hair_head)

		middle_front.paste(hair_torso, (int(((self.width) - hair_torso.size[0])/2) + int(torso_offset[0]*self.width), int(((self.height) - hair_torso.size[1])/2 + torso_wobble[1]) + int(torso_offset[1]*self.height)), hair_torso)

		middle_front.paste(left_arm, (0 + int(arm_offset[0]*self.width), int(((self.height) - left_arm.size[1])/2 + arm_wobble[1]) + + int(arm_offset[1]*self.height)), left_arm)

		middle_front.paste(right_arm, (int(((self.width) - right_arm.size[0])) - int(arm_offset[0]*self.width), int(((self.height) - right_arm.size[1])/2 + arm_wobble[1]) + int(arm_offset[1]*self.height)), right_arm)

		middle_front.paste(left_leg, (int((self.width)/2 - left_leg.size[0]) + int(leg_offset[0]*self.width), int(((self.height) - left_leg.size[1])) + int(leg_offset[1]*self.height)), left_leg)

		middle_front.paste(right_leg, (int((self.width)/2) - int(leg_offset[0]*self.width), int(((self.height) - right_leg.size[1])) + int(leg_offset[1]*self.height)), right_leg)

		middle_front.paste(torso, (int(((self.width) - torso.size[0])/2) + int(torso_offset[0]*self.width), int(((self.height) - torso.size[1])/2 + torso_wobble[1]) + int(torso_offset[1]*self.height)), torso)

		middle_front.paste(head, (int(((self.width) - head.size[0])/2) + int(head_offset[0]*self.width), 0 + head_wobble[1] + int(head_offset[1]*self.height)), head)

		self.middle_front = middle_front
		#middle_front.save("Generated_Sprites/" + str(i) + "_Fused_Front_Middle"  + ".png","PNG")


		# Making left front facing
		left_front = Image.new("RGBA", ((self.width), (self.height)), (0,0,0,0))

		left_front.paste(hair_head, (int(((self.width) - hair_head.size[0])/2 + head_wobble[0]) + int(head_offset[0]*self.width), 0 + int(head_offset[1]*self.height)), hair_head)

		left_front.paste(left_arm, (0 + arm_wobble[0] + int(arm_offset[0]*self.width), int(((self.height) - left_arm.size[1])/2) + int(arm_offset[1]*self.height)), left_arm)

		left_front.paste(hair_torso, (int(((self.width) - hair_torso.size[0])/2 + torso_wobble[0]) + int(torso_offset[0]*self.width), int(((self.height) - hair_torso.size[1])/2) + int(torso_offset[1]*self.height)), hair_torso)

		left_front.paste(right_leg, (int((self.width)/2 + leg_wobble[0]) - int(leg_offset[0]*self.width), int(((self.height) - right_leg.size[1])) - leg_wobble[1] + int(leg_offset[1]*self.height)), right_leg)

		left_front.paste(torso, (int(((self.width) - torso.size[0])/2 + torso_wobble[0]) + int(torso_offset[0]*self.width), int(((self.height) - torso.size[1])/2) + int(torso_offset[1]*self.height)), torso)

		left_front.paste(left_leg, (int((self.width)/2 - left_leg.size[0]) + int(leg_offset[0]*self.width), int(((self.height) - left_leg.size[1])) + int(leg_offset[1]*self.height)), left_leg)

		left_front.paste(head, (int(((self.width) - head.size[0])/2 + head_wobble[0]) + int(head_offset[0]*self.width), 0 + int(head_offset[1]*self.height)), head)

		left_front.paste(right_arm, (int(((self.width) - right_arm.size[0])) - arm_wobble[0] - int(arm_offset[0]*self.width), int(((self.height) - right_arm.size[1])/2) + int(arm_offset[1]*self.height)), right_arm)

		self.left_front = left_front

		#left_front.save("Generated_Sprites/" + str(i) + "_Fused_Front_Left"  + ".png","PNG")


		# Making right front facing
		right_front = Image.new("RGBA", ((self.width), (self.height)), (0,0,0,0))

		right_front.paste(hair_head, (int(((self.width) - hair_head.size[0])/2) - head_wobble[0]+ int(head_offset[0]*self.width), 0 + int(head_offset[1]*self.height)), hair_head)

		right_front.paste(right_arm, (int(((self.width) - right_arm.size[0])) - arm_wobble[0] - int(arm_offset[0]*self.width), int(((self.height) - right_arm.size[1])/2) + int(arm_offset[1]*self.height)), right_arm)

		right_front.paste(hair_torso, (int(((self.width) - hair_torso.size[0])/2 - torso_wobble[0]) + int(torso_offset[0]*self.width), int(((self.height) - hair_torso.size[1])/2) + int(torso_offset[1]*self.height)), hair_torso)

		right_front.paste(left_leg, (int((self.width)/2 - left_leg.size[0] - leg_wobble[0]) + int(leg_offset[0]*self.width), int(((self.height) - left_leg.size[1])) - leg_wobble[1] + int(leg_offset[1]*self.height)), left_leg)

		right_front.paste(torso, (int(((self.width) - torso.size[0])/2 - torso_wobble[0]) + int(torso_offset[0]*self.width), int(((self.height) - torso.size[1])/2) + int(torso_offset[1]*self.height)), torso)

		right_front.paste(right_leg, (int((self.width)/2) - int(leg_offset[0]*self.width), int(((self.height) - right_leg.size[1])) + int(leg_offset[1]*self.height)), right_leg)

		right_front.paste(head, (int(((self.width) - head.size[0])/2) -head_wobble[0] + int(head_offset[0]*self.width), 0 + int(head_offset[1]*self.height)), head)

		right_front.paste(left_arm, (0 + arm_wobble[0] + int(arm_offset[0]*self.width), int(((self.height) - left_arm.size[1])/2) + int(arm_offset[1]*self.height)), left_arm)

		self.right_front = right_front
		#right_front.save("Generated_Sprites/" + str(i) + "_Fused_Front_Right"  + ".png","PNG")



		front_triad.paste(left_front, (0,0), left_front)
		front_triad.paste(middle_front, ((self.width), 0), middle_front)
		front_triad.paste(right_front, ((self.width)*2, 0), right_front)
		#front_triad.save("Generated_Sprites/" + str(i) + "_Front_Triad" + "_" + str((self.width)) + "x" + str((self.height)) + ".png","PNG")





		############################################################
		#														   #
		#				Making Back Facing Sprites 				   #
		#														   #
		############################################################




		back_triad = Image.new("RGBA", ((self.width)*3, (self.height)), (0,0,0,0))


		# Making middle back facing
		middle_back = Image.new("RGBA", ((self.width), (self.height)), (0,0,0,0))

		middle_back.paste(back_left_arm, (0 + int(arm_offset[0]*self.width), int(((self.height) - back_left_arm.size[1])/2) + arm_wobble[1] + int(arm_offset[1]*self.height)), back_left_arm)

		middle_back.paste(back_right_arm, (int(((self.width) - back_right_arm.size[0]) - int(arm_offset[0]*self.width)), int(((self.height) - back_right_arm.size[1])/2) + arm_wobble[1] + int(arm_offset[1]*self.height)), back_right_arm)

		middle_back.paste(back_left_leg, (int((self.width)/2 - back_left_leg.size[0]) + int(leg_offset[0]*self.width), int(((self.height) - back_left_leg.size[1])) + int(leg_offset[1]*self.height)), back_left_leg)

		middle_back.paste(back_right_leg, (int((self.width)/2) - int(leg_offset[0]*self.width), int(((self.height) - back_right_leg.size[1])) + int(leg_offset[1]*self.height)), back_right_leg)

		middle_back.paste(back_head, (int(((self.width) - back_head.size[0])/2) - int(head_offset[0]*self.width), 0 + head_wobble[1] + int(head_offset[1]*self.height)), back_head)

		middle_back.paste(back_torso, (int(((self.width) - back_torso.size[0])/2) - int(torso_offset[0]*self.width), int(((self.height) - back_torso.size[1])/2 + torso_wobble[1]) + int(torso_offset[1]*self.height)), back_torso)

		middle_back.paste(hair_torso, (int(((self.width) - hair_torso.size[0])/2) - int(torso_offset[0]*self.width), int(((self.height) - hair_torso.size[1])/2 + torso_wobble[1]) + int(torso_offset[1]*self.height)), hair_torso)

		middle_back.paste(hair_head, (int(((self.width) - hair_head.size[0])/2) - int(head_offset[0]*self.width), 0 + head_wobble[1] + int(head_offset[1]*self.height)), hair_head)

		self.middle_back = middle_back

		#middle_back.save("Generated_Sprites/" + str(i) + "_Fused_Front_Middle"  + ".png","PNG")


		# Making left back facing
		left_back = Image.new("RGBA", ((self.width), (self.height)), (0,0,0,0))

		left_back.paste(back_right_arm, (int(((self.width) - back_right_arm.size[0])) - arm_wobble[0] - int(arm_offset[0]*self.width), int(((self.height) - back_right_arm.size[1])/2) + int(arm_offset[1]*self.height)), back_right_arm)

		left_back.paste(back_right_leg, (int((self.width)/2) + leg_wobble[0] - int(leg_offset[0]*self.width), int(((self.height) - back_right_leg.size[1])) - leg_wobble[1] + int(leg_offset[1]*self.height)), back_right_leg)

		left_back.paste(back_left_leg, (int((self.width)/2 - back_left_leg.size[0]) + int(leg_offset[0]*self.width), int(((self.height) - back_left_leg.size[1])) + int(leg_offset[1]*self.height)), back_left_leg)

		left_back.paste(back_head, (int(((self.width) - back_head.size[0])/2 -head_wobble[0]) - int(head_offset[0]*self.width), 0 + int(head_offset[1]*self.height)), back_head)

		left_back.paste(back_torso, (int(((self.width) - back_torso.size[0])/2 - torso_wobble[0]) - int(torso_offset[0]*self.width), int(((self.height) - back_torso.size[1])/2) + int(torso_offset[1]*self.height)), back_torso)

		left_back.paste(hair_torso, (int(((self.width) - hair_torso.size[0])/2 - torso_wobble[0]) - int(torso_offset[0]*self.width), int(((self.height) - hair_torso.size[1])/2) + int(torso_offset[1]*self.height)), hair_torso)

		left_back.paste(hair_head, (int(((self.width) - hair_head.size[0])/2 -head_wobble[0]) - int(head_offset[0]*self.width), 0 + int(head_offset[1]*self.height)), hair_head)

		left_back.paste(back_left_arm, (0 + arm_wobble[0] + int(arm_offset[0]*self.width), int(((self.height) - back_left_arm.size[1])/2) + int(arm_offset[1]*self.height)), back_left_arm)

		self.left_back = left_back
		

		#left_back.save("Generated_Sprites/" + str(i) + "_Fused_Front_Left"  + ".png","PNG")


		# Making right back facing
		right_back = Image.new("RGBA", ((self.width), (self.height)), (0,0,0,0))

		right_back.paste(back_left_arm, (0 + arm_wobble[0] + int(arm_offset[0]*self.width), int(((self.height) - back_left_arm.size[1])/2) + int(arm_offset[1]*self.height)), back_left_arm)

		right_back.paste(back_left_leg, (int((self.width)/2 - back_left_leg.size[0]) - leg_wobble[0] + int(leg_offset[0]*self.width), int(((self.height) - back_left_leg.size[1])) - leg_wobble[1] + int(leg_offset[1]*self.height) ), back_left_leg)

		right_back.paste(back_right_leg, (int((self.width)/2) - int(leg_offset[0]*self.width), int(((self.height) - back_right_leg.size[1])) + int(leg_offset[1]*self.height)), back_right_leg)

		right_back.paste(back_head, (int(((self.width) - back_head.size[0])/2 + head_wobble[0]) - int(head_offset[0]*self.width), 0 + int(head_offset[1]*self.height)), back_head)

		right_back.paste(back_torso, (int(((self.width) - back_torso.size[0])/2 + torso_wobble[0] - int(torso_offset[0]*self.width)), int(((self.height) - back_torso.size[1])/2) + int(torso_offset[1]*self.height)), back_torso)

		right_back.paste(hair_torso, (int(((self.width) - hair_torso.size[0])/2 + torso_wobble[0]) - int(torso_offset[0]*self.width), int(((self.height) - hair_torso.size[1])/2) + int(torso_offset[1]*self.height)), hair_torso)

		right_back.paste(hair_head, (int(((self.width) - hair_head.size[0])/2 + head_wobble[0]) - int(head_offset[0]*self.width), 0 + int(head_offset[1]*self.height)), hair_head)

		right_back.paste(back_right_arm, (int(((self.width) - back_right_arm.size[0])) - arm_wobble[0] - int(arm_offset[0]*self.width), int(((self.height) - back_right_arm.size[1])/2) + int(arm_offset[1]*self.height)), back_right_arm)

		self.right_back = right_back
		#right_back.save("Generated_Sprites/" + str(i) + "_Fused_Front_Right"  + ".png","PNG")

		back_triad.paste(left_back, (0,0), left_back)
		back_triad.paste(middle_back, ((self.width), 0), middle_back)
		back_triad.paste(right_back, ((self.width)*2, 0), right_back)
		#back_triad.save("Generated_Sprites/" + str(i) + "_Back_Triad" + "_" + str((self.width)) + "x" + str((self.height))  + ".png","PNG")






		# ############################################################
		# #														   #
		# #				Making Side Facing Sprites 				   #
		# #														   #
		# ############################################################





		side_triad = Image.new("RGBA", ((self.width)*3, (self.height)), (0,0,0,0))


		# Making middle side facing
		middle_side = Image.new("RGBA", ((self.width), (self.height)), (0,0,0,0))

		middle_side.paste(side_right_arm.rotate(side_arm_angles[0], PIL.Image.NEAREST, expand = 1), (int(((self.width)/2 - side_right_arm.rotate(side_arm_angles[0], PIL.Image.NEAREST, expand = 1).size[0]/2)), int(((self.height) - side_right_arm.rotate(side_arm_angles[0], PIL.Image.NEAREST, expand = 1).size[1])/2) + arm_wobble[1] + int(arm_offset[1]*self.height)), side_right_arm.rotate(side_arm_angles[0], PIL.Image.NEAREST, expand = 1))

		middle_side.paste(side_left_leg, (int((self.width)/2 - side_left_leg.size[0]/2), int(((self.height) - side_left_leg.size[1])) + int(leg_offset[1]*self.height)), side_left_leg)

		middle_side.paste(side_torso, (int(((self.width) - side_torso.size[0])/2), int(((self.height) - side_torso.size[1])/2 + torso_wobble[1]) + int(torso_offset[1]*self.height)), side_torso)

		middle_side.paste(side_head, (int(((self.width) - side_head.size[0])/2), 0 + head_wobble[1] + int(head_offset[1]*self.height)), side_head)

		middle_side.paste(side_left_leg, (int((self.width)/2 - side_left_leg.size[0]/2), int(((self.height) - side_left_leg.size[1])) + int(leg_offset[1]*self.height)), side_left_leg)

		middle_side.paste(side_right_arm.rotate(side_arm_angles[0], PIL.Image.NEAREST, expand = 1), (int(((self.width)/2 - side_right_arm.rotate(side_arm_angles[0], PIL.Image.NEAREST, expand = 1).size[0]/2)), int(((self.height) - side_right_arm.rotate(side_arm_angles[0], PIL.Image.NEAREST, expand = 1).size[1])/2) + arm_wobble[1] + int(arm_offset[1]*self.height)), side_right_arm.rotate(side_arm_angles[0], PIL.Image.NEAREST, expand = 1))

		self.middle_side = middle_side


		# Making left side facing
		left_side = Image.new("RGBA", ((self.width), (self.height)), (0,0,0,0))

		left_side.paste(side_right_arm.rotate(side_arm_angles[1], PIL.Image.NEAREST, expand = 1), (int(((self.width)/2 - side_right_arm.rotate(side_arm_angles[1], PIL.Image.NEAREST, expand = 1).size[0]/2)) - side_arm_wobble[0], int(((self.height) - side_right_arm.rotate(side_arm_angles[1], PIL.Image.NEAREST, expand = 1).size[1])/2) + int(arm_offset[1]*self.height)), side_right_arm.rotate(side_arm_angles[1], PIL.Image.NEAREST, expand = 1))

		left_side.paste(side_left_leg, (int((self.width)/2 - side_left_leg.size[0]/2) + side_leg_wobble[0], int(((self.height) - side_left_leg.size[1])) - side_leg_wobble[1] + int(leg_offset[1]*self.height)), side_left_leg)

		left_side.paste(side_torso, (int(((self.width) - side_torso.size[0])/2), int(((self.height) - side_torso.size[1])/2) + int(torso_offset[1]*self.height)), side_torso)

		left_side.paste(side_head, (int(((self.width) - side_head.size[0])/2), 0 + int(head_offset[1]*self.height)), side_head)

		left_side.paste(side_left_leg, (int((self.width)/2 - side_left_leg.size[0]/2) - side_leg_wobble[0], int(((self.height) - side_left_leg.size[1]) - side_leg_wobble[1]) + int(leg_offset[1]*self.height)), side_left_leg)

		left_side.paste(side_right_arm.rotate(side_arm_angles[2], PIL.Image.NEAREST, expand = 1), (int(((self.width)/2 - side_right_arm.rotate(side_arm_angles[2], PIL.Image.NEAREST, expand = 1).size[0]/2)) + side_arm_wobble[0], int(((self.height) - side_right_arm.rotate(side_arm_angles[2], PIL.Image.NEAREST, expand = 1).size[1])/2) + int(arm_offset[1]*self.height)), side_right_arm.rotate(side_arm_angles[2], PIL.Image.NEAREST, expand = 1))

		self.left_side = left_side
		#left_back.save("Generated_Sprites/" + str(i) + "_Fused_Front_Left"  + ".png","PNG")


		# Making right side facing
		right_side = Image.new("RGBA", ((self.width), (self.height)), (0,0,0,0))

		right_side.paste(side_right_arm.rotate(side_arm_angles[2], PIL.Image.NEAREST, expand = 1), (int(((self.width)/2 - side_right_arm.rotate(side_arm_angles[2], PIL.Image.NEAREST, expand = 1).size[0]/2)) + side_arm_wobble[0], int(((self.height) - side_right_arm.rotate(side_arm_angles[2], PIL.Image.NEAREST, expand = 1).size[1])/2) + int(arm_offset[1]*self.height)), side_right_arm.rotate(side_arm_angles[2], PIL.Image.NEAREST, expand = 1))

		right_side.paste(side_left_leg, (int((self.width)/2 - side_left_leg.size[0]/2) - side_leg_wobble[0], int(((self.height) - side_left_leg.size[1]) - side_leg_wobble[1]) + int(leg_offset[1]*self.height)), side_left_leg)

		right_side.paste(side_torso, (int(((self.width) - side_torso.size[0])/2), int(((self.height) - side_torso.size[1])/2) + int(torso_offset[1]*self.height)), side_torso)

		right_side.paste(side_head, (int(((self.width) - side_head.size[0])/2), 0 + int(head_offset[1]*self.height)), side_head)

		right_side.paste(side_left_leg, (int((self.width)/2 - side_left_leg.size[0]/2) + side_leg_wobble[0], int(((self.height) - side_left_leg.size[1])- side_leg_wobble[1]) + int(leg_offset[1]*self.height)), side_left_leg)

		right_side.paste(side_right_arm.rotate(side_arm_angles[1], PIL.Image.NEAREST, expand = 1), (int(((self.width)/2 - side_right_arm.rotate(side_arm_angles[1], PIL.Image.NEAREST, expand = 1).size[0]/2)) - side_arm_wobble[0], int(((self.height) - side_right_arm.rotate(side_arm_angles[1], PIL.Image.NEAREST, expand = 1).size[1])/2) + int(arm_offset[1]*self.height)), side_right_arm.rotate(side_arm_angles[1], PIL.Image.NEAREST, expand = 1))

		self.right_side = right_side

		#right_back.save("Generated_Sprites/" + str(i) + "_Fused_Front_Right"  + ".png","PNG")

		side_triad.paste(left_side, (0,0), left_side)
		side_triad.paste(middle_side, ((self.width), 0), middle_side)
		side_triad.paste(right_side, ((self.width)*2, 0), right_side)
		#back_triad.save("Generated_Sprites/" + str(i) + "_Back_Triad" + "_" + str((self.width)) + "x" + str((self.height))  + ".png","PNG")




		sprite_sheet_list = []

		#Making Sprite Sheet

		for size_multiplication in self.size_multiplication_list:

			sprite_sheet = Image.new("RGBA", ((self.width*size_multiplication)*3, (self.height*size_multiplication)*4), (0,0,0,0))

			sprite_sheet.paste(front_triad.resize((self.width*size_multiplication*3,self.height*size_multiplication),resample=Image.NEAREST), (0,0), front_triad.resize((self.width*size_multiplication*3,self.height*size_multiplication),resample=Image.NEAREST))

			sprite_sheet.paste(side_triad.resize((self.width*size_multiplication*3,self.height*size_multiplication),resample=Image.NEAREST), (0,(self.height*size_multiplication)), side_triad.resize((self.width*size_multiplication*3,self.height*size_multiplication),resample=Image.NEAREST))

			sprite_sheet.paste(ImageOps.mirror(side_triad).resize((self.width*size_multiplication*3,self.height*size_multiplication),resample=Image.NEAREST), (0,(self.height*size_multiplication)*2), ImageOps.mirror(side_triad).resize((self.width*size_multiplication*3,self.height*size_multiplication),resample=Image.NEAREST))

			sprite_sheet.paste(back_triad.resize((self.width*size_multiplication*3,self.height*size_multiplication),resample=Image.NEAREST), (0,(self.height*size_multiplication)*3), back_triad.resize((self.width*size_multiplication*3,self.height*size_multiplication),resample=Image.NEAREST))

			sprite_sheet_list.append(sprite_sheet)

			if save_name is not None:

				sprite_sheet.save("Generated_Sprites/" + str(save_name) + "_Sprite_Sheet" + "_" + str((self.width*size_multiplication)) + "x" + str((self.height*size_multiplication)) + ".png","PNG")

				animation_sequence = [0,1,2,1,0,1,2,1]

				direction_sequence = [0,1,3,2]

				gif_images = []

				for direction in direction_sequence:
					for seq in animation_sequence:
						image = sprite_sheet.crop((seq*self.width*size_multiplication, direction*self.height*size_multiplication, seq*self.width*size_multiplication + self.width*size_multiplication, direction*self.height*size_multiplication + self.height*size_multiplication))
						gif_images.append(image)


				# gif_images = [left_front.resize((self.width*size_multiplication,self.height*size_multiplication),resample=Image.NEAREST), middle_front.resize((self.width*size_multiplication,self.height*size_multiplication),resample=Image.NEAREST), right_front.resize((self.width*size_multiplication,self.height*size_multiplication),resample=Image.NEAREST), middle_front.resize((self.width*size_multiplication,self.height*size_multiplication),resample=Image.NEAREST), left_back.resize((self.width*size_multiplication,self.height*size_multiplication),resample=Image.NEAREST), middle_back.resize((self.width*size_multiplication,self.height*size_multiplication),resample=Image.NEAREST), right_back.resize((self.width*size_multiplication,self.height*size_multiplication),resample=Image.NEAREST), middle_back.resize((self.width*size_multiplication,self.height*size_multiplication),resample=Image.NEAREST)]

				gif_images[0].save("Generated_Sprites/" + "animated_" + str(i) + "_" + str((self.width*size_multiplication)) + "x" + str((self.height*size_multiplication)) + ".gif",
	               save_all=True,
	               append_images=gif_images[1:],
	               duration=250,
	               loop=0)
		self.sprite_sheet_list = sprite_sheet_list
		return sprite_sheet_list

	def save(self, save_name, gif_speed_delay = 250, size_multiplication = 20):

		sheety = self.sprite_sheet_list[0].resize((self.sprite_sheet_list[0].width*size_multiplication, self.sprite_sheet_list[0].height*size_multiplication),resample=Image.NEAREST)
		sheety.save(save_name + ".png","PNG")

		animation_sequence = [0,1,2,1,0,1,2,1]

		direction_sequence = [0,1,3,2]

		gif_images = []

		for direction in direction_sequence:
			for seq in animation_sequence:
				image = sheety.crop((seq*self.width*size_multiplication, direction*self.height*size_multiplication, seq*self.width*size_multiplication + self.width*size_multiplication, direction*self.height*size_multiplication + self.height*size_multiplication))
				gif_images.append(image)


		#gif_images = [self.left_front.resize((self.width*size_multiplication,self.height*size_multiplication),resample=Image.NEAREST), self.middle_front.resize((self.width*size_multiplication,self.height*size_multiplication),resample=Image.NEAREST), self.right_front.resize((self.width*size_multiplication,self.height*size_multiplication),resample=Image.NEAREST), self.middle_front.resize((self.width*size_multiplication,self.height*size_multiplication),resample=Image.NEAREST), self.left_back.resize((self.width*size_multiplication,self.height*size_multiplication),resample=Image.NEAREST), self.middle_back.resize((self.width*size_multiplication,self.height*size_multiplication),resample=Image.NEAREST), self.right_back.resize((self.width*size_multiplication,self.height*size_multiplication),resample=Image.NEAREST), self.middle_back.resize((self.width*size_multiplication,self.height*size_multiplication),resample=Image.NEAREST)]

		gif_images[0].save(save_name + ".gif",
           save_all=True,
           append_images=gif_images[1:],
           duration = gif_speed_delay,
           loop=0)




# NPC_Limb_Movement_Generator()

# exit()

if __name__ == "__main__":


	size_multiplication_list = [1, 10]

	sprity = NPC_Humanoid_Generator(32, 32, size_multiplication_list)

	for i in range(10):

		print(i)
		sprity.makeNPC(save_name = i)








































