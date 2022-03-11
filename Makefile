CC = nvcc
CFLAGS = -lcurand 
ARCHFLAG = -gencode arch=compute_50,code=sm_50 -Wno-deprecated-gpu-targets

TARGET1 = ./final_attack/offline/maxterm_mining/GPU_version_5/att1
TARGET2 = ./final_attack/offline/maxterm_mining/GPU_version_5/only_gpu/att1
TARGET3 = ./final_attack/offline/maxterm_mining/GPU_version_6/att1
TARGET4 = ./final_attack/offline/maxterm_mining/GPU_version_6/only_gpu/att1
TARGET5 = ./final_attack/offline/maxterm_mining/GPU_version_7/att1
TARGET6 = ./final_attack/offline/maxterm_mining/GPU_version_7/only_gpu/att1
TARGET7 = ./final_attack/offline/CubeValid/GPU_version_5/att1
TARGET8 = ./final_attack/offline/CubeValid/GPU_version_5/only_gpu/att1
TARGET9 = ./final_attack/offline/CubeValid/GPU_version_6/att1
TARGET10 = ./final_attack/offline/CubeValid/GPU_version_6/only_gpu/att1
TARGET11 = ./final_attack/offline/CubeValid/GPU_version_7/att1
TARGET12 = ./final_attack/offline/CubeValid/GPU_version_7/only_gpu/att1
TARGET13 = ./final_attack/offline/superpoly_rec/att1
TARGET14 = ./final_attack/offline/superpoly_rec/only_gpu/att1
TARGET15 = ./final_attack/online/att1
TARGET16 = ./final_attack/validate/att1
TARGET17 = ./final_attack/validate/only_gpu/att1


all: $(TARGET1) $(TARGET2) $(TARGET3) $(TARGET4) $(TARGET5) $(TARGET6) $(TARGET7) $(TARGET8) $(TARGET9) $(TARGET10) $(TARGET11) $(TARGET12) $(TARGET13) $(TARGET14) $(TARGET15) $(TARGET16) $(TARGET17)

$(TARGET1): $(TARGET1).cu
	$(CC) $(CFLAGS) $(ARCHFLAG) -o $(TARGET1) $(TARGET1).cu

$(TARGET2): $(TARGET2).cu
	$(CC) $(CFLAGS) $(ARCHFLAG) -o $(TARGET2) $(TARGET2).cu

$(TARGET3): $(TARGET3).cu
	$(CC) $(CFLAGS) $(ARCHFLAG) -o $(TARGET3) $(TARGET3).cu

$(TARGET4): $(TARGET4).cu
	$(CC) $(CFLAGS) $(ARCHFLAG) -o $(TARGET4) $(TARGET4).cu

$(TARGET5): $(TARGET5).cu
	$(CC) $(CFLAGS) $(ARCHFLAG) -o $(TARGET5) $(TARGET5).cu

$(TARGET6): $(TARGET6).cu
	$(CC) $(CFLAGS) $(ARCHFLAG) -o $(TARGET6) $(TARGET6).cu

$(TARGET7): $(TARGET7).cu
	$(CC) $(CFLAGS) $(ARCHFLAG) -o $(TARGET7) $(TARGET7).cu

$(TARGET8): $(TARGET8).cu
	$(CC) $(CFLAGS) $(ARCHFLAG) -o $(TARGET8) $(TARGET8).cu

$(TARGET9): $(TARGET9).cu
	$(CC) $(CFLAGS) $(ARCHFLAG) -o $(TARGET9) $(TARGET9).cu

$(TARGET10): $(TARGET10).cu
	$(CC) $(CFLAGS) $(ARCHFLAG) -o $(TARGET10) $(TARGET10).cu

$(TARGET11): $(TARGET11).cu
	$(CC) $(CFLAGS) $(ARCHFLAG) -o $(TARGET11) $(TARGET11).cu

$(TARGET12): $(TARGET12).cu
	$(CC) $(CFLAGS) $(ARCHFLAG) -o $(TARGET12) $(TARGET12).cu

$(TARGET13): $(TARGET13).cu
	$(CC) $(CFLAGS) $(ARCHFLAG) -o $(TARGET13) $(TARGET13).cu

$(TARGET14): $(TARGET14).cu
	$(CC) $(CFLAGS) $(ARCHFLAG) -o $(TARGET14) $(TARGET14).cu

$(TARGET15): $(TARGET15).cu
	$(CC) $(CFLAGS) $(ARCHFLAG) -o $(TARGET15) $(TARGET15).cu

$(TARGET16): $(TARGET16).cu
	$(CC) $(CFLAGS) $(ARCHFLAG) -o $(TARGET16) $(TARGET16).cu

$(TARGET17): $(TARGET17).cu
	$(CC) $(CFLAGS) $(ARCHFLAG) -o $(TARGET17) $(TARGET17).cu

clean:
	$(RM) $(TARGET1)
	$(RM) $(TARGET2)
	$(RM) $(TARGET3)
	$(RM) $(TARGET4)
	$(RM) $(TARGET5)
	$(RM) $(TARGET6)	
	$(RM) $(TARGET7)
	$(RM) $(TARGET8)	
	$(RM) $(TARGET9)
	$(RM) $(TARGET10)	
	$(RM) $(TARGET11)
	$(RM) $(TARGET12)	
	$(RM) $(TARGET13)
	$(RM) $(TARGET14)	
	$(RM) $(TARGET15)
	$(RM) $(TARGET16)	
	$(RM) $(TARGET17)