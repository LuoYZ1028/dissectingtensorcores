# $(MAKE)、$(MAKE_FLAGS)是预设的make命令和flags

BASE_DIR := $(shell pwd) # 获取相对路径
BIN_DIR := $(BASE_DIR)/bin
# SUB_DIRS      = $(wildcard ubench/*/*/)
# SUB_DIRS      = $(wildcard ubench/ldmatrix/*/)
# SUB_DIRS      = $(wildcard ubench/mma/*/)
# SUB_DIRS		= $(wildcard ubench/wmma_load/*/)
# SUB_DIRS		= $(wildcard ubench/wmma/*/)
SUB_DIRS 		= $(wildcard ubench/wmma/wmma_m16n16k16_fp16/)

# 此处:%=做了一个添加前缀的操作
SUB_DIRS_ALL    = $(SUB_DIRS:%=all-%)
# $(warning $(SUB_DIRS_ALL))
SUB_DIRS_CLEAN  = $(SUB_DIRS:%=clean-%)

all: create_dir $(SUB_DIRS_ALL)	

clean: delete_dir $(SUB_DIRS_CLEAN)

# make -c 表示到子目录下执行子目录的 Makefile
$(SUB_DIRS_ALL):
	$(MAKE) $(MAKE_FLAGS) -C $(@:all-%=%)

# 同理，这里表示到子目录下执行 make clean 操作
$(SUB_DIRS_CLEAN):
	$(MAKE) $(MAKE_FLAGS) -C $(@:clean-%=%) clean

create_dir:
	mkdir -p $(BIN_DIR)	

delete_dir:
	cd $(BIN_DIR); rm -f *.app;


	
	
