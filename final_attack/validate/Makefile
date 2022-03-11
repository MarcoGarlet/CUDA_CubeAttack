CC = nvcc
CFLAGS = -lcurand 
ARCHFLAG = -gencode arch=compute_50,code=sm_50 -Wno-deprecated-gpu-targets
TARGET = att1

all: $(TARGET)
$(TARGET): $(TARGET).cu
	$(CC) $(CFLAGS) -o $(TARGET) $(TARGET).cu
clean:
	$(RM) $(TARGET)