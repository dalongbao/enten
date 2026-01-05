EMCC = emcc
CFLAGS = -O2
SRC = main.c fish.c simulator.c
HEADERS = fish.h simulator.h
OUT = build/index.html

all: $(OUT)

EXPORTS = ["_main","_set_action","_get_x","_get_y","_get_angle","_get_vx","_get_vy","_get_food_count","_get_observation","_drop_food","_get_tail_phase","_get_tail_amplitude","_get_body_curve","_get_left_pectoral","_get_right_pectoral","_get_food_positions","_get_screen_w","_get_screen_h","_malloc","_free","_get_fish_count_sim","_set_action_for_fish","_get_x_for_fish","_get_y_for_fish","_get_angle_for_fish","_get_observation_for_fish","_get_tail_phase_for_fish","_get_tail_amplitude_for_fish","_get_body_curve_for_fish","_get_left_pectoral_for_fish","_get_right_pectoral_for_fish"]

$(OUT): $(SRC) $(HEADERS) shell.html renderer.js
	mkdir -p build
	$(EMCC) $(CFLAGS) -o $(OUT) $(SRC) -s EXPORTED_FUNCTIONS='$(EXPORTS)' -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap","getValue","setValue"]' --shell-file shell.html
	cp renderer.js build/

clean:
	rm -rf build

.PHONY: all clean
