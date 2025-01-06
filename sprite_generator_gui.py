from tkinter import *
from PIL import ImageTk, Image
import npc_sprite_generator
import random
from scipy.stats import truncnorm

IMAGE_SIZE = (300,400)
GIF_SIZE = (200,200)

SIZE_VALUE_LIST = [16,32,64]

ANIMATION_SEQUENCE = [0,1,2,1,0,1,2,1]
DIRECTION_SEQUENCE = [0,1,3,2]

SIZE_SLIDER_PARAMETER_LIST = [
("Torso Dim. X:", (0,100), 60, 60),
("Torso Dim. Y:", (0,100), 60, 60),
("Arm Dim. X:", (0,100), 25, 25),
("Arm Dim. Y:", (0,100), 40, 40),
("Leg Dim. X:", (0,100), 25, 25),
("Leg Dim. Y:", (0,100), 40, 40),
("Head Dim. X:", (0,100), 50, 50),
("Head Dim. Y:", (0,100), 50, 50),
]


ANIMATION_SLIDER_PARAMETER_LIST = [
("Head Wobble X:", (-10,10), 0, 0),
("Head Wobble Y:", (-10,10), 2, 2),
("Torso Wobble X:", (-10,10), 0, 0),
("Torso Wobble Y:", (-10,10), 1, 1),
("Arm Wobble X:", (-10,10), 1, 1),
("Arm Wobble Y:", (-10,10), 1, 1),
("Side Arm Wobble X:", (-10,10), 2, 2),
("Side Arm Wobble Y:", (-10,10), 0, 0),
("Leg Wobble X:", (-10,10), 0, 0),
("Leg Wobble Y:", (-10,10), 1, 1),
("Side Leg Wobble X:", (-10,10), 2, 2),
("Side Leg Wobble Y:", (-10,10), 0, 0),
("Side Arm Angles X:", (0,360), 335, 335),
("Side Arm Angles Y:", (0,360), 25, 25),
]


POSITIONING_SLIDER_PARAMETER_LIST = [
("Torso Offset X:", (-50,50), 0, 0),
("Torso Offset Y:", (-50,50), 0, 0),
("Arm Offset X:", (-50,50), 20, 20),
("Arm Offset Y:", (-50,50), 0, 0),
("Leg Offset X:", (-50,50), 0, 0),
("Leg Offset Y:", (-50,50), 0, 0),
("Head Offset X:", (-50,50), 0, 0),
("Head Offset Y:", (-50,50), 0, 0),
]


SPRITE_SIZE = 32

size_slider_list = []
size_label_list = []
animation_slider_list = []
animation_label_list = []
positioning_slider_list = []
positioning_label_list = []
colour_slider_list = []
colour_label_list = []
colour_input_list = []
random_colours_label = None
palette_colours_label = None
colour_slider = None

GENERATOR = None


random_sd = 5


def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


def generate_new_sprite():

	global test
	global gif_images_list
	global GENERATOR
	global save_path
	global english_nouns
	global english_adjectives

	GENERATOR = npc_sprite_generator.NPC_Humanoid_Generator(size_slider.get(), size_slider.get(), [1])

	sprite_sheet = GENERATOR.makeNPC(
		torso_dim = (size_slider_list[0].get()/100, size_slider_list[1].get()/100),
		arm_dim = (size_slider_list[2].get()/100, size_slider_list[3].get()/100),
		leg_dim = (size_slider_list[4].get()/100, size_slider_list[5].get()/100),
		head_dim = (size_slider_list[6].get()/100, size_slider_list[7].get()/100),
		head_wobble = (animation_slider_list[0].get(), animation_slider_list[1].get()),
		torso_wobble = (animation_slider_list[2].get(), animation_slider_list[3].get()),
		arm_wobble = (animation_slider_list[4].get(), animation_slider_list[5].get()),
		side_arm_wobble = (animation_slider_list[6].get(), animation_slider_list[7].get()),
		leg_wobble = (animation_slider_list[8].get(), animation_slider_list[9].get()),
		side_leg_wobble = (animation_slider_list[10].get(), animation_slider_list[11].get()),
		side_arm_angles = (0, animation_slider_list[12].get(), animation_slider_list[13].get()),
		torso_offset = (positioning_slider_list[0].get()/100, positioning_slider_list[1].get()/100),
		arm_offset = (positioning_slider_list[2].get()/100, positioning_slider_list[3].get()/100),
		leg_offset = (positioning_slider_list[4].get()/100, positioning_slider_list[5].get()/100),
		head_offset = (positioning_slider_list[6].get()/100, positioning_slider_list[7].get()/100),
		)[0]
	test = ImageTk.PhotoImage(sprite_sheet.resize(IMAGE_SIZE, resample=Image.NEAREST))
	sprite_sheet_image.configure(image = test)

	gif_images_list = []

	for direction in DIRECTION_SEQUENCE:
		for seq in ANIMATION_SEQUENCE:
			image = ImageTk.PhotoImage(sprite_sheet.crop((seq*size_slider.get(), direction*size_slider.get(), seq*size_slider.get() + size_slider.get(), direction*size_slider.get() + size_slider.get())).resize(GIF_SIZE, resample=Image.NEAREST))
			gif_images_list.append(image)

	new_name = random.choice(english_adjectives) + "_" + random.choice(english_nouns) 

	save_path.delete("1.0", END)
	save_path.insert(END, "./" + "Generated_Sprites" + "/" + new_name)

def size_value_check(value):
	newvalue = min(SIZE_VALUE_LIST, key=lambda x:abs(x-float(value)))
	size_slider.set(newvalue)

def animate_GIF(frame_number):
	if frame_number >= len(gif_images_list):
		frame_number = 0
	gif_images.config(image=gif_images_list[frame_number]) 
	window.after(gif_vel_slider.get(), animate_GIF, frame_number+1)

def reset_parameters():

	for sliddy, paramy in zip(size_slider_list, SIZE_SLIDER_PARAMETER_LIST):
		sliddy.set(paramy[2])
	for sliddy, paramy in zip(animation_slider_list, ANIMATION_SLIDER_PARAMETER_LIST):
		sliddy.set(paramy[2])
	for sliddy, paramy in zip(positioning_slider_list, POSITIONING_SLIDER_PARAMETER_LIST):
		sliddy.set(paramy[2])

def random_parameters():
	global random_sd

	for sliddy, paramy in zip(size_slider_list, SIZE_SLIDER_PARAMETER_LIST):
		sliddy.set(get_truncated_normal(mean=paramy[2], sd=20, low=paramy[1][0], upp=paramy[1][1]).rvs())
		#sliddy.set(random.randint(paramy[1][0], paramy[1][1]))
	for sliddy, paramy in zip(animation_slider_list, ANIMATION_SLIDER_PARAMETER_LIST):
		#sliddy.set(random.randint(paramy[1][0], paramy[1][1]))
		sliddy.set(get_truncated_normal(mean=paramy[2], sd=1, low=paramy[1][0], upp=paramy[1][1]).rvs())
	for sliddy, paramy in zip(positioning_slider_list, POSITIONING_SLIDER_PARAMETER_LIST):
		sliddy.set(get_truncated_normal(mean=paramy[2], sd=2, low=paramy[1][0], upp=paramy[1][1]).rvs())
		#sliddy.set(random.randint(paramy[1][0], paramy[1][1]))

	generate_new_sprite()


def generate_size_sliders():

	global size_slider_list
	global size_label_list
	global size_slider
	global size_label
	global SPRITE_SIZE


	# Size Slider

	size_label = Label(text="Sprite Size:", fg="#0D3A5F", bg="#F4E7D3")
	size_label.grid(row=2, column=0, padx=5, sticky = W)
	size_slider = Scale(window, from_=min(SIZE_VALUE_LIST), to=max(SIZE_VALUE_LIST), command=size_value_check, orient=HORIZONTAL, fg="#0D3A5F", bg="#F4E7D3", length = 200)
	size_slider.set(SPRITE_SIZE)
	size_slider.grid(row=2, column=1, columnspan = 2, padx=5, sticky = W)

	size_slider_list = []
	size_label_list = []

	cur_row = 4
	for parameters in SIZE_SLIDER_PARAMETER_LIST:
		par_label = Label(text=parameters[0], fg="#0D3A5F", bg="#F4E7D3")
		par_label.grid(row=cur_row, column=0, padx=5, sticky = W)
		size_label_list.append(par_label)
		par_slider = Scale(window, from_=parameters[1][0], to=parameters[1][1], orient=HORIZONTAL, fg="#0D3A5F", bg="#F4E7D3", length = 200)
		par_slider.set(parameters[2])
		par_slider.grid(row=cur_row, column=1, columnspan = 2, padx=5, sticky = W)
		cur_row += 1
		size_slider_list.append(par_slider)

def generate_positioning_sliders():

	global positioning_slider_list
	global positioning_label_list


	positioning_slider_list = []
	positioning_label_list = []

	cur_row = 4
	for parameters in POSITIONING_SLIDER_PARAMETER_LIST:
		par_label = Label(text=parameters[0], fg="#0D3A5F", bg="#F4E7D3")
		par_label.grid(row=cur_row, column=0, padx=5, sticky = W)
		positioning_label_list.append(par_label)
		par_slider = Scale(window, from_=parameters[1][0], to=parameters[1][1], orient=HORIZONTAL, fg="#0D3A5F", bg="#F4E7D3", length = 200)
		par_slider.set(parameters[2])
		par_slider.grid(row=cur_row, column=1, columnspan = 2, padx=5, sticky = W)
		cur_row += 1
		positioning_slider_list.append(par_slider)



def generate_animation_sliders():

	global animation_slider_list
	global animation_label_list


	# Size Slider

	animation_slider_list = []
	animation_label_list = []

	cur_row = 3
	for parameters in ANIMATION_SLIDER_PARAMETER_LIST:
		par_label = Label(text=parameters[0], fg="#0D3A5F", bg="#F4E7D3")
		par_label.grid(row=cur_row, column=0, padx=5, sticky = W)
		animation_label_list.append(par_label)
		par_slider = Scale(window, from_=parameters[1][0], to=parameters[1][1], orient=HORIZONTAL, fg="#0D3A5F", bg="#F4E7D3", length = 200)
		par_slider.set(parameters[2])
		par_slider.grid(row=cur_row, column=1, columnspan = 2, padx=5, sticky = W)
		cur_row += 1
		animation_slider_list.append(par_slider)

def colour_check(value):

	if value == "0":
		activate_random_colours()
		disable_custom_palette()
	else:
		disable_random_colours()
		activate_custom_palette()



def generate_colour_inputs():

	global colour_input_list
	global colour_label_list
	global random_colours_label
	global palette_colours_label
	global colour_slider

	random_colours_label = Label(text='Harmonious Colours', fg="#0D3A5F", bg="#F4E7D3")
	random_colours_label.grid(row=3, column=1, padx=5, sticky = W)

	palette_colours_label = Label(text='Custom Palette', fg="#0D3A5F", bg="#F4E7D3")
	palette_colours_label.grid(row=4, column=1, padx=5, sticky = W)
	
	colour_slider = Scale(window, from_=0, to=1, command=colour_check, orient=VERTICAL, fg="#0D3A5F", bg="#F4E7D3", length = 60, showvalue=0)
	colour_slider.set(0)
	colour_slider.grid(row=3, column=0, rowspan = 2, padx=5, sticky = E)


	colour_input_list = []
	colour_label_list = []

	num_colours = 10

	cur_row = 5
	for col_num in range(num_colours):

		par_label = Label(text="Colour " + str(col_num + 1), fg="#0D3A5F", bg="#F4E7D3")
		par_label.grid(row=cur_row, column=0, padx=5, sticky = W)
		input1 = Text(window, height = 1, width = 20)
		input1.grid(row=cur_row, column=1, columnspan = 2, padx=5, sticky = W)
		cur_row += 1
		colour_input_list.append(input1)
		colour_label_list.append(par_label)


def activate_random_colours():
	
	global random_colours_label

	random_colours_label.configure(state = NORMAL)
	


def disable_random_colours():

	global random_colours_label

	random_colours_label.configure(state = "disabled")



def activate_custom_palette():


	global colour_input_list
	global colour_label_list
	global palette_colours_label

	palette_colours_label.configure(state = NORMAL)

	for colly in colour_input_list:
		colly.configure(state = NORMAL)

	for lolly in colour_label_list:
		lolly.configure(state = NORMAL)

	pass
	
def disable_custom_palette():

	global colour_input_list
	global colour_label_list
	global palette_colours_label

	palette_colours_label.configure(state = "disabled")

	for colly in colour_input_list:
		colly.configure(state = "disabled")

	for lolly in colour_label_list:
		lolly.configure(state = "disabled")



def hide_colour_inputs():

	global colour_input_list
	global colour_label_list
	global random_colours_label
	global palette_colours_label
	global colour_slider

	random_colours_label.grid_remove()
	palette_colours_label.grid_remove()
	colour_slider.grid_remove()

	for sliddy in colour_input_list:
		sliddy.grid_remove()

	for labby in colour_label_list:
		labby.grid_remove()


def show_colour_inputs():

	global colour_input_list
	global colour_label_list
	global random_colours_label
	global palette_colours_label
	global colour_slider


	hide_size_sliders()
	hide_animation_sliders()
	hide_positioning_sliders()

	random_colours_label.grid()
	palette_colours_label.grid()
	colour_slider.grid()

	for sliddy in colour_input_list:
		sliddy.grid()

	for labby in colour_label_list:
		labby.grid()




def hide_animation_sliders():

	global animation_slider_list
	global animation_label_list

	for sliddy in animation_slider_list:
		sliddy.grid_remove()

	for labby in animation_label_list:
		labby.grid_remove()


def show_animation_sliders():

	global animation_slider_list
	global animation_label_list

	hide_size_sliders()
	hide_colour_inputs()
	hide_positioning_sliders()

	for sliddy in animation_slider_list:
		sliddy.grid()
		sliddy.configure(state = NORMAL)

	for labby in animation_label_list:
		labby.grid()


def hide_size_sliders():

	global size_slider_list
	global size_slider
	global size_label_list
	global size_label

	size_slider.grid_remove()
	size_label.grid_remove()
	for sliddy in size_slider_list:
		sliddy.grid_remove()

	for labby in size_label_list:
		labby.grid_remove()

def show_size_sliders():

	global size_slider_list
	global size_slider
	global size_label_list
	global size_label

	

	size_slider.grid()
	size_label.grid()
	for sliddy in size_slider_list:
		sliddy.grid()

	for labby in size_label_list:
		labby.grid()

	hide_animation_sliders()
	hide_colour_inputs()
	hide_positioning_sliders()


def hide_positioning_sliders():

	global positioning_slider_list
	global positioning_label_list

	for sliddy in positioning_slider_list:
		sliddy.grid_remove()

	for labby in positioning_label_list:
		labby.grid_remove()

def show_positioning_sliders():

	global positioning_slider_list
	global positioning_label_list

	for sliddy in positioning_slider_list:
		sliddy.grid()

	for labby in positioning_label_list:
		labby.grid()

	hide_animation_sliders()
	hide_colour_inputs()
	hide_size_sliders()



def place_holder_function():
	pass

def save_sprite_sheet():
	global save_path
	global GENERATOR
	global gif_vel_slider

	speed = gif_vel_slider.get()
	path = save_path.get("1.0",END)

	GENERATOR.save(path.replace("\n", ""), gif_speed_delay = speed)




with open("./Words/english-nouns.txt") as my_file:
	english_nouns = my_file.read().split("\n")

with open("./Words/english-adjectives.txt") as my_file:
	english_adjectives = my_file.read().split("\n")


window = Tk()

window.configure(bg="#F4E7D3")
window.title("Sprite Maker 3000")


# GIF Velocity Slider

Label(text="GIF Delay:", fg="#0D3A5F", bg="#F4E7D3").grid(row=25, column=3, padx=5, sticky = W)
gif_vel_slider = Scale(window, from_=1, to=500, orient=HORIZONTAL, fg="#0D3A5F", bg="#F4E7D3")
gif_vel_slider.set(250)
gif_vel_slider.grid(row=25, column=4, padx=5, sticky = W)


#Size Sliders

generate_size_sliders()

generate_animation_sliders()
hide_animation_sliders()

generate_colour_inputs()
activate_random_colours()
disable_custom_palette()
hide_colour_inputs()


generate_positioning_sliders()
hide_positioning_sliders()


# Save

save_button = Button(window, text='Save!', fg="#0D3A5F", highlightbackground="#F4E7D3", command=save_sprite_sheet)
save_button.grid(row=32, column=5, columnspan = 1, sticky = E)

save_label = Label(text="Save Path:", fg="#0D3A5F", bg="#F4E7D3")
save_label.grid(row=32, column=4, sticky = W)
save_path = Text(window, height = 1, width = 40)
save_path.insert(END, "./")
save_path.grid(row=32, column=4, pady= 5, columnspan = 3, padx=15)

# Sprite Sheet Image

sprite_sheet_image = Label(fg="#0D3A5F", bg="#F4E7D3")
sprite_sheet_image.grid(row=2, column=5, pady=5, padx=5, rowspan = 30, sticky = E)


gif_images = Label(fg="#0D3A5F", bg="#F4E7D3")
gif_images.grid(row=2, column=3, pady=5, padx=5, rowspan = 30, columnspan = 2, sticky = E)


generate_new_sprite()

animate_GIF(0)

# Generate Button

generate_button = Button(window, text='Generate!', fg="#0D3A5F", highlightbackground="#F4E7D3", command=generate_new_sprite)
generate_button.grid(row=32, column=1, columnspan = 1, pady=5, padx=5)

# Reset Button

reset_button = Button(window, text='Reset All Parameters', fg="#0D3A5F", highlightbackground="#F4E7D3", command=reset_parameters)
reset_button.grid(row=32, column=0, columnspan = 1, pady=5, padx=5)

# Randomize Button

random_button = Button(window, text='Soft Randomize', fg="#0D3A5F", highlightbackground="#F4E7D3", command=random_parameters)
random_button.grid(row=32, column=2, columnspan = 2, pady=5, padx=5)


# Size Button

size_button = Button(window, text='Size & Sizes', fg="#0D3A5F", highlightbackground="#F4E7D3", command=show_size_sliders)
size_button.grid(row=1, column=0, columnspan = 1, pady=5, padx=5)

# Positioning

positioning_button = Button(window, text='Positioning', fg="#0D3A5F", highlightbackground="#F4E7D3", command=show_positioning_sliders)
positioning_button.grid(row=1, column=1, columnspan = 1, pady=5, padx=5)

# Animation

animation_button = Button(window, text='Animation', fg="#0D3A5F", highlightbackground="#F4E7D3", command=show_animation_sliders)
animation_button.grid(row=1, column=2, columnspan = 1, pady=5, padx=5, sticky = E)

# # Colours

# colours_button = Button(window, text='Colours', fg="#0D3A5F", highlightbackground="#F4E7D3", command=show_colour_inputs)
# colours_button.grid(row=1, column=3, columnspan = 1, pady=5, padx=5, sticky = E)

# # Algorithm

# algorithm_button = Button(window, text='Algorithm', fg="#0D3A5F", highlightbackground="#F4E7D3", command=place_holder_function)
# algorithm_button.grid(row=1, column=4, columnspan = 1, pady=5, padx=5, sticky = E)




# Calling Main Loop

window.mainloop()

