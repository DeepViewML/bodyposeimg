OBJS := bodyposeimg.o
LIBS := -lvaal

%.o : %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS) 

bodyposeimg: $(OBJS)
	dpkg -L libvaal
	$(CC) -o $@ $^ $(LDFLAGS) $(LIBS)


install: bodyposeimg
	mkdir -p $(WORKDIR)
	cp bodyposeimg $(WORKDIR)/


clean:
	rm -f *.o
	rm -f bodyposeimg
