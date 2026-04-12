CC = gcc
CFLAGS = -O3 -shared

SO_DIR = connect4
SOURCES = $(SO_DIR)/minimax_c.c $(SO_DIR)/solver_c.c
TARGETS = $(SOURCES:.c=.so)

all: $(TARGETS)

$(SO_DIR)/%.so: $(SO_DIR)/%.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)

.PHONY: all clean
