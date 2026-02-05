CC = gcc
CFLAGS = -Wall -I.
CFLAGS += -I $(HOME)/opt/raylib-5.5_linux_amd64/include

LDLIBS = -lrt -ldl -lX11 -lpthread -lm

RAYLIB_STATIC_LIB = $(HOME)/opt/raylib-5.5_linux_amd64/lib/libraylib.a

TARGET = gblurr
SRCS = gblurr.c

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(SRCS) -o $(TARGET) $(CFLAGS) $(RAYLIB_STATIC_LIB) $(LDLIBS)

clean:
	rm -f $(TARGET)
