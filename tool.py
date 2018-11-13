#headers
import json 
import configparser
import numpy as np
import math
import sys
import csv


outFile = open('vgg16_results.csv', 'w')
writer = csv.writer(outFile)
writer.writerow(["layer_name","inputFeatureMapPosition","iUpSamplingRatio","iChannels","iRows","iCols","iBytes"
		,"outFeatureMapPosition","oDownSamplingRatio","oChannels","oRows","oCols","oBytes"
		,"filterMapPosition","fUpSamplingRatio","fIn","fRows","fCols","fBytes"
		,"iDataMoveTime","oDataMoveTime","fDataMoveTime","totalDataMoveTime"
		,"mComputeTime","vComputeTime","totalComputeTime"
		,"serialComputeForLayer","parallelComputeForLayer"])

#read hardware model from config file. Read everything as double.
config = configparser.ConfigParser()
config.read('hardware.config')
hardwareConfig = config["HardwareConfig"]

memoryBandwidth = float(hardwareConfig["memoryBandwidth"]) * 1e9
internalMemorySize = float(hardwareConfig["internalMemorySize"]) * 1e6 #converting MB to bytes
availableMemorySize = internalMemorySize
primitiveM = float(hardwareConfig["primitiveM"])
primitiveN = float(hardwareConfig["primitiveN"])
primitiveK = float(hardwareConfig["primitiveK"])
primitiveMNK = primitiveM * primitiveN * primitiveK
parallelMatMulPrimitives = float(hardwareConfig["parallelMatMulPrimitives"])
matMulPrimitivesCompletionRate = float(hardwareConfig["matMulPrimitivesCompletionRate"])
vectorPrimitiveSize = float(hardwareConfig["vectorPrimitiveSize"])
parallelVectorPrimitives = float(hardwareConfig["parallelVectorPrimitives"])
vectorPrimitivesCompletionRate = float(hardwareConfig["vectorPrimitivesCompletionRate"])

#print function
def printStats(layer_name,inputFeatureMapPosition,iUpSamplingRatio,iChannels,iRows,iCols,iBytes
		,outFeatureMapPosition,oDownSamplingRatio,oChannels,oRows,oCols,oBytes
		,filterMapPosition,fUpSamplingRatio,fIn,fRows,fCols,fBytes
		,iDataMoveTime,oDataMoveTime,fDataMoveTime,totalDataMoveTime
		,mComputeTime,vComputeTime,totalComputeTime
		,serialComputeForLayer,parallelComputeForLayer):
	print("layer_name ",layer_name)
	print("----------------------------------------------------------------------------")
	
	print("input feature map position",inputFeatureMapPosition)
	print("input feature map upsampling ratio",iUpSamplingRatio)
	print("no of input feature maps",iChannels)
	print("no of input feature maps rows",iRows)
	print("no of input feature maps cols",iCols)
	print("no of input feature maps bytes",iBytes)
	print("output feature map position",outFeatureMapPosition)
	print("output feature map downsampling ratio",oDownSamplingRatio)
	print("no of output feature maps",oChannels)
	print("no of output feature maps rows",oRows)
	print("no of output feature maps cols",oCols)
	print("no of output feature maps bytes",oBytes)
	print("filter map position",filterMapPosition)
	print("filter map upsampling ratio",fUpSamplingRatio)
	print("filter grouping",fIn)
	print("no of filter rows",fRows)
	print("no of filter cols",fCols)
	print("no of filter bytes",fBytes)
	print("input tensor data movement time ",iDataMoveTime)
	print("output tensor data movement time ",oDataMoveTime)
	print("filter map data movement time ",fDataMoveTime)
	print("total data movement time ",totalDataMoveTime)
	print("matrix compute time ",mComputeTime)
	print("vector compute time ",vComputeTime)
	print("total compute time ",totalComputeTime)
	print("serial compute for layer ",serialComputeForLayer)
	print("parallel compute for layer ",parallelComputeForLayer)
	print("----------------------------------------------------------------------------")
#
with open('vgg16_pyformat.json') as data_file:
    modelJson = json.load(data_file)
model = json.loads(modelJson)
layers = model["config"]["layers"]
inputLayer = model["config"]["input_layers"][0][0]
outputLayer = model["config"]["input_layers"][0][0]
print(len(layers))
input_shape = []
iChannels = 0
iBatchSize = 1
iRows = 0
iCols = 0
iBytes = 0
iUpSamplingRatio = 0
oChannels = 0
oRows = 0
oCols = 0
oBytes = 0
oDownSamplingRatio = 0
fOut = 0
fIn = 0
fRows = 0
fCols = 0
fBytes = 0
fUpSamplingRatio = 0
inputFeatureMapPosition = 1 #external
outFeatureMapPosition = 0 #internal
filterMapPosition = 1 #external
iDataMoveTime = 0
oDataMoveTime = 0
fDataMoveTime = 0
iDataMoveTimeForNetwork = 0
oDataMoveTimeForNetwork = 0
fDataMoveTimeForNetwork = 0
totalDataMoveTimeForNetwork = 0

totalMatrixComputeTime = 0
totalVectorComputeTime = 0
totalComputeTimeForNetwork = 0

serialComputeForLayer = 0
parallelComputeForLayer = 0
serialComputeForNetwork = 0
parallelComputeForNetwork = 0

layers_details = {}


for layer in layers:
	layer_type = layer["class_name"]
	layer_name = layer["name"]
	inbound_nodes = layer["inbound_nodes"]
	
	mComputeTime = 0
	vComputeTime = 0
	iUpSamplingRatio = 0
	fUpSamplingRatio = 0
	oDownSamplingRatio = 0
	fOut = 0
	fIn = 0
	fRows = 0
	fCols = 0
	fBytes = 0
	filterMapPosition = 1
	config = layer["config"]
	
	#assign input from previous layer. This works only if the current layer has only one parent
	if layer_type.lower() != "inputlayer" and layer_type.lower() != "add" and layer_type.lower() != "concatenate":
		prevLayerName = inbound_nodes[0][0][0]
		prevLayer = layers_details[prevLayerName]
		iChannels = prevLayer["oChannels"]
		iRows = prevLayer["oRows"]
		iCols = prevLayer["oCols"]
		iBytes = iBatchSize*iChannels * iRows * iCols #correct logic to include batch size for tensor
		inputFeatureMapPosition = prevLayer["outFeatureMapPosition"]
	
	if layer_type.lower() == "inputlayer" and layer_name == inputLayer:
		input_shape = layer["config"]["batch_input_shape"]
		iChannels = input_shape[3]
		iRows = input_shape[1]
		iCols = input_shape[2]
		iBytes = iBatchSize*iChannels * iRows * iCols #correct logic to include batch size for tensor
		oChannels = iChannels
		oRows = iRows
		oCols = iCols
		oBytes = iBytes
		
		
		
	elif layer_type.lower() == "zeropadding2d":
		padding = config["padding"]
		tP = padding[0][0]
		bP = padding[0][1]
		lP = padding[1][0]
		rP = padding[1][1]
		oChannels = iChannels
		oRows = iRows + tP + bP
		oCols = iCols + lP + rP
		oBytes = iBatchSize*oChannels *oRows * oCols	
		
	elif layer_type.lower() == "batchnormalization":
		oChannels = iChannels
		oRows = iRows
		oCols = iCols
		oBytes = iBytes	
	
		
	elif layer_type.lower() == "conv2d":		
		
		fOut = config["filters"]
		fIn = iChannels
		fRows = config["kernel_size"][0]
		fCols = config["kernel_size"][1]
		fBytes = fOut * fIn * fRows * fCols
		sR = config["strides"][0]
		sC = config["strides"][1]
		padding = config["padding"]
		if padding.lower() == "same":
			pR = fRows - 1
			pC = fCols - 1
			fUpSamplingRatio = 0
		elif padding.lower() == "valid":
			pR = 0
			pC = 0
		oDownSamplingRatio = sR	#check how to implement this when sR!=sC
		oChannels = fOut
		oRows = math.ceil(((iRows + pR - fRows +1)/sR))
		oCols = math.ceil(((iCols + pC - fCols +1)/sC))	
		oBytes = iBatchSize*oChannels *oRows * oCols 
		#MNK - (output dimension * no of input and output * input dimension)
		
		mMNK = oRows * oCols  * oChannels * iChannels * fRows * fCols 
		print(mMNK)
		mPrimitiveMAC = mMNK / (primitiveMNK * parallelMatMulPrimitives)
		mComputeTime = mPrimitiveMAC / matMulPrimitivesCompletionRate
		print(mComputeTime)
		
		
	elif layer_type.lower() == "depthwiseconv2d":		
		
		fOut = iChannels * config["depth_multiplier"]
		fIn = iChannels
		fRows = config["kernel_size"][0]
		fCols = config["kernel_size"][1]
		fBytes = fOut * fIn * fRows * fCols
		sR = config["strides"][0]
		sC = config["strides"][1]
		padding = config["padding"]
		if padding.lower() == "same":
			pR = fRows - 1
			pC = fCols - 1
			fUpSamplingRatio = 0
		elif padding.lower() == "valid":
			pR = 0
			pC = 0
		oDownSamplingRatio = sR	#check how to implement this when sR!=sC
		oChannels = fOut
		oRows = math.ceil(((iRows + pR - fRows +1)/sR))
		oCols = math.ceil(((iCols + pC - fCols +1)/sC))	
		oBytes = iBatchSize*oChannels *oRows * oCols 
		#MNK - (output dimension * no of input and output * input dimension)
		mMNK = oRows * oCols  * oChannels * iChannels * fRows * fCols
		mPrimitiveMAC = mMNK / (primitiveMNK * (parallelMatMulPrimitives))
		mComputeTime = mPrimitiveMAC / matMulPrimitivesCompletionRate
		
		
		
	elif layer_type.lower() == "activation" or layer_type.lower() == "relu" :
		oChannels = iChannels
		oRows = iRows
		oCols = iCols
		oBytes = iBytes
		
	elif layer_type.lower() == "add":
		inBound = inbound_nodes[0]
		iBytes = 0
		for prevLayerName in inBound:			
			prevLayer = layers_details[prevLayerName[0]]
			iChannels = prevLayer["oChannels"]
			iRows = prevLayer["oRows"]
			iCols = prevLayer["oCols"]
			iBytes += iBatchSize * iChannels * iRows * iCols #correct logic to include batch size for tensor
			inputFeatureMapPosition = prevLayer["outFeatureMapPosition"]
		oChannels = iChannels
		oRows = iRows
		oCols = iCols
		oBytes = iBatchSize*oChannels * oRows * oCols
		
	elif layer_type.lower() == "concatenate":
		inBound = inbound_nodes[0]
		oChannels = 0
		iBytes = 0
		iChannels = 0
		for prevLayerName in inBound:			
			prevLayer = layers_details[prevLayerName[0]]
			iChannelsTemp = prevLayer["oChannels"]
			iChannels += iChannelsTemp #netscope concats all inputs and tells 256 as input channle. check
			iRows = prevLayer["oRows"]
			iCols = prevLayer["oCols"]
			iBytes += iBatchSize * iChannelsTemp * iRows * iCols #correct logic to include batch size for tensor
			inputFeatureMapPosition = prevLayer["outFeatureMapPosition"]
		oChannels = iChannels
		oRows = iRows
		oCols = iCols
		oBytes = iBatchSize*oChannels * oRows * oCols
		
		
	elif layer_type.lower() == "maxpooling2d" or layer_type.lower() == "averagepooling2d":
	
		sR = config["strides"][0]
		sC = config["strides"][1]		
		kR = config["pool_size"][0]
		kC = config["pool_size"][1]
		padding = config["padding"]
		if padding.lower() == "same":
			pR = kR - 1
			pC = kC - 1
			fUpSamplingRatio = 0
		elif padding.lower() == "valid":
			pR = 0
			pC = 0
		oDownSamplingRatio = sR
		oRows = math.ceil(((iRows + pR - kR)+1)/sR)
		oCols = math.ceil(((iCols + pC - kC)+1)/sC)	
		oBytes = iBatchSize*oChannels *oRows * oCols
		
		
	elif layer_type.lower() == "flatten":
		oChannels = iBatchSize * iChannels * iRows * iCols
		oRows = 1
		oCols = 1
		
	elif layer_type.lower() == "globalaveragepooling2d":
		oChannels = iChannels
		oRows = 1
		oCols = 1
		
	elif layer_type.lower() == "reshape":
		target_shape = config["target_shape"]
		if len(target_shape)==3:
			oChannels = target_shape[2]
			oRows = target_shape[0]
			oCols = target_shape[1]
		elif len(target_shape) == 1:
			oChannels = target_shape[0]
			oRows = 1
			oCols = 1
			
		oBytes = iBatchSize*oChannels * oRows * oCols
	
	elif layer_type.lower() == "dropout":
		oChannels = iChannels
		oRows = iRows
		oCols = iCols
		oBytes = iBytes
	
	elif layer_type.lower() == "dense":
	
		fRows = oChannels
		fCols = iChannels
		fBytes = oChannels * iChannels
		oChannels = config["units"]
		oRows = iRows
		oCols = iCols
		oBytes = iBatchSize*oChannels *oRows * oCols
		vMK = oChannels * iChannels * 1
		vPrimitiveMAC = vMK / (vectorPrimitiveSize * parallelVectorPrimitives)
		vComputeTime = vPrimitiveMAC / vectorPrimitivesCompletionRate
		
	
	else:
		print("unknown layer",layer_type)
		sys.exit()
		
	if availableMemorySize-oBytes >= 0:
		outFeatureMapPosition = 0
		availableMemorySize = internalMemorySize - oBytes
	else:
		outFeatureMapPosition = 1
	
	if layer_type.lower() != "conv2d" and layer_type.lower() != "depthwiseconv2d" and layer_type.lower() != "dense" and layer_type.lower() != "activation" and layer_type.lower() != "relu":
		vMK = max(iBytes,oBytes)
		vPrimitiveMAC = vMK / (vectorPrimitiveSize * parallelVectorPrimitives)
		vComputeTime = vPrimitiveMAC / vectorPrimitivesCompletionRate
		
	
	iDataMoveTime = inputFeatureMapPosition * iBytes / memoryBandwidth
	iDataMoveTimeForNetwork += iDataMoveTime
	oDataMoveTime = outFeatureMapPosition * oBytes / memoryBandwidth
	oDataMoveTimeForNetwork += oDataMoveTime
	fDataMoveTime = filterMapPosition * fBytes / memoryBandwidth
	fDataMoveTimeForNetwork += fDataMoveTime
	totalDataMoveTime = iDataMoveTime + oDataMoveTime + fDataMoveTime
	totalComputeTime = mComputeTime + vComputeTime
	totalMatrixComputeTime += mComputeTime
	totalVectorComputeTime += vComputeTime
	serialComputeForLayer = totalDataMoveTime + totalComputeTime
	parallelComputeForLayer = max(totalDataMoveTime,totalComputeTime)
	serialComputeForNetwork += serialComputeForLayer
	parallelComputeForNetwork += parallelComputeForLayer
	'''
	printStats(layer_name,inputFeatureMapPosition,iUpSamplingRatio,iChannels,iRows,iCols,iBytes
		,outFeatureMapPosition,oDownSamplingRatio,oChannels,oRows,oCols,oBytes
		,filterMapPosition,fUpSamplingRatio,fIn,fRows,fCols,fBytes
		,iDataMoveTime,oDataMoveTime,fDataMoveTime,totalDataMoveTime
		,mComputeTime,vComputeTime,totalComputeTime
		,serialComputeForLayer,parallelComputeForLayer)
	'''
	writer.writerow([layer_name,inputFeatureMapPosition,iUpSamplingRatio,iChannels,iRows,iCols,iBytes
		,outFeatureMapPosition,oDownSamplingRatio,oChannels,oRows,oCols,oBytes
		,filterMapPosition,fUpSamplingRatio,fIn,fRows,fCols,fBytes
		,iDataMoveTime,oDataMoveTime,fDataMoveTime,totalDataMoveTime
		,mComputeTime,vComputeTime,totalComputeTime
		,serialComputeForLayer,parallelComputeForLayer])

	#add this to layer_info
	layer_info = {}
	layer_info["oChannels"] = oChannels
	layer_info["oRows"] = oRows
	layer_info["oCols"] = oCols
	layer_info["oBytes"] = oBytes
	layer_info["outFeatureMapPosition"] = outFeatureMapPosition
	layers_details[layer_name] = layer_info
	''' 
	#works only for sequential 
	#set output as input of next layer
	iChannels = oChannels
	iRows = oRows
	iCols = oCols
	iBytes = iChannels * iRows * iCols #correct logic to include batch size for tensor
	inputFeatureMapPosition = outFeatureMapPosition
	'''

#total net stats
totalDataMoveTimeForNetwork = iDataMoveTimeForNetwork + oDataMoveTimeForNetwork + fDataMoveTimeForNetwork
totalComputeTimeForNetwork = totalMatrixComputeTime + totalVectorComputeTime
writer.writerow(["iDataMoveTimeForNetwork","oDataMoveTimeForNetwork","fDataMoveTimeForNetwork","totalDataMoveTimeForNetwork"
	,"totalMatrixComputeTime","totalVectorComputeTime","totalComputeTimeForNetwork"
	,"serialComputeForNetwork","parallelComputeForNetwork"])
writer.writerow([iDataMoveTimeForNetwork,oDataMoveTimeForNetwork,fDataMoveTimeForNetwork,totalDataMoveTimeForNetwork
	,totalMatrixComputeTime,totalVectorComputeTime,totalComputeTimeForNetwork
	,serialComputeForNetwork,parallelComputeForNetwork])