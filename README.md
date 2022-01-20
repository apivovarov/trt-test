# trt-test

### Download models
```
wget -i models.txt
```

### Extract all models
```
ls *.tar.gz | xargs -I  {} tar zxf {}

# delete model archives
rm *.tar.gz 
```

### Convert to TRT
```
./convert.sh
```

### Test models performance
```
./time.sh
```
