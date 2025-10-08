package org.example;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.util.Random;

public class HBCApproximate {
    private static final Random sssrandom = new Random();

    // 存储数据的map <（u，v)，时间戳集合）> 时间戳可重复
    public Map<Edge, Set<Integer>> edgeToTimestamps=new HashMap<>();
    // 边数量
    public int edgeNum;
    // 总weight
    public int totalWeight = 0;
    // 最远遍历距离
    public int maxDistance=0;
    // 成功样本数量
    public int successSample = 0;
    // 成功尺度
    public int successScale = 4;

    // 存储点的邻居节点
    public Map<Integer, HashSet<Integer>> neighbors=new HashMap<Integer,HashSet<Integer>>();
    // 存在的snapshot
    public HashSet<Integer> timestamps = new HashSet<>();

    // 存储节点集
    public HashSet<Integer> nodes = new HashSet<>();
    // 存储节点集
    public ArrayList<Integer> nodesArray = new ArrayList<>();

    // 节点的度中心性
    Map<Integer, Integer> degreeWeights = new HashMap<>();
    // 归一化重要性
    TreeMap<Integer, Integer> cumulativeWeights = new TreeMap<>();

    // 节点度加权提取的已删除节点
    ArrayList<Integer> deletedNodes = new ArrayList<>();
    // 记录采样结果Vw
    public Map<Integer, List<Double>> Vw = new HashMap<>();
    // 记录遍历得到路径条数
//    public int pathsum = 0;
    // 记录加入的边数
    public int edgesum = 0;
    public long samplingTime = 0;
    public long calculateTime = 0;


    public static void main(String[] args) {
        String filePath = "D:/code/superuser.txt";
        HBCApproximate hbc = new HBCApproximate(filePath);
        hbc.CalculateTBC(0.03,0.1,100000);
    }

    public HBCApproximate(String fileName) {
        BufferedReader reader = null;
        int alreadyLines = 0;
        int alreadyNum = 0;
        int circles = 0;
        try {
            // 初始化BufferedReader以读取文件
            reader = new BufferedReader(new FileReader(fileName));
            String line;
//            reader.readLine();
            // 循环读取文件的每一行
            while ((line = reader.readLine()) != null) {
                alreadyLines++;
                if(alreadyLines>=1000000){
                    alreadyNum++;
                    alreadyLines = 0;
                    System.out.println("已导入 " + alreadyNum +" 百万行数据；");
                }
                // 通过空格分隔每行数据为多个部分
                String[] tokens = line.split("\\s+");
                int timestamp = Integer.parseInt(tokens[0]);
                int start = Integer.parseInt(tokens[1]) ;
                int end = Integer.parseInt(tokens[2]);

                // 目前表示有向图
                // 将时间戳和标志添加到目标数据结构中
                timestamps.add(timestamp);
                if(start==end){
//                    System.out.println("有独立环");
                    circles++;
                    continue;
                }

                degreeWeights.compute(start, (k, v) -> (v == null) ? 1 : v + 1);

                edgeToTimestamps.compute(new Edge(start, end), (edge, list) -> {
                    if (list == null) {
                        list = new HashSet<>();
                    }
                    if(!list.contains(timestamp)){
                        edgesum++;
                    }
                    list.add(timestamp);
                    return list;
                });
                // 将邻居加入到点
                neighbors.compute(start, (node, set) -> {
                    if (set == null) {
                        set = new HashSet<>();
                    }
                    set.add(end);
                    return set;
                });
                // 有向图需要自己添加节点
                nodes.add(start);
                nodes.add(end);
            }
            // 更新点集
            nodesArray.addAll(nodes);

            for(int node:this.nodesArray){
                Vw.put(node,new ArrayList<>());
            }
            updateWeight(degreeWeights);
        } catch (FileNotFoundException e) {
            // 如果文件未找到，抛出运行时异常
            throw new RuntimeException(e);
        } catch (IOException e) {
            // 如果发生IO异常，抛出运行时异常
            throw new RuntimeException(e);
        } finally {
            // 确保BufferedReader在使用后被正确关闭
            if (reader != null){
                try {
                    reader.close();
                } catch (IOException e) {
                    // 在关闭过程中发生的IO异常同样抛出运行时异常
                    throw new RuntimeException(e);
                }
            }
        }
    }

    // 重载函数之单点采样
    // 计算：pathsum更新；
    public Boolean CalculateSSSPModify(int source, int interval_s, int interval_t){
        long currentStartTime = System.currentTimeMillis();
        int level =0;
        int stampSum  = this.timestamps.size()+1;
        int num = 1;

        // 每个节点u保存长度为T的时间戳数组；
        Map<Integer,int[]> dist=new HashMap<>();
        // BFS遍历队列；
        Queue<Integer> currentAppearance = new LinkedList<>();
        Queue<ArrayList<Integer>> currentTime = new LinkedList<>();
        // 距离递增的顶点实例队列，直接记录id
        Stack<Integer> increaseNode = new Stack<>();

        // 记录中间结果
        // 记录节点外观，外观id-节点id;
        Map<Integer,Integer> lifespans = new HashMap<>();
        // 记录节点外观，外观id-lifespan;
        Map<Integer,ArrayList<Integer>> lifespansStamp = new HashMap<>();
        // 记录节点名下id，节点id-外观id;
        Map<Integer,ArrayList<Integer>> nodeAppearance = new HashMap<>();
        // 记录前驱；
        Map<Integer,Integer> predecessor = new HashMap<>();
        // 记录节点外观路径数量
        Map<Integer,Integer> pathnum = new HashMap<>();
        // 记录分母，即到达节点总路径数量
        Map<Integer,Integer> pathSum = new HashMap<>();
        // 记录拜访记录
        Map<Integer,ArrayList<Integer>> visited = new HashMap<>();

        // 要检查source是否存在
        // 初始化
        for(int node:this.nodesArray){
            int[] sourceArray = new int[stampSum];
            Arrays.fill(sourceArray,-1);
            dist.put(node,sourceArray);
        }

        // 初始化source访问距离
        int[] sourceArray = new int[stampSum];
        Arrays.fill(sourceArray,0);
        dist.put(source,sourceArray);
        // 初始化source访问时间
        ArrayList<Integer> initial = new ArrayList<>();
        for (int i = interval_s; i <interval_t+1; i++) {
            initial.add(i);
        }
        visited.put(source,initial);
        // 初始化SSSP访问序列
        currentAppearance.add(num);
        currentTime.add(initial);
        // 初始化source的节点外观
        lifespans.put(num,source);
        lifespansStamp.put(num,initial);
        nodeAppearance.computeIfAbsent(source,k->new ArrayList<>()).add(num);
        // 初始化source的路径条数
        pathnum.put(num,1);
        pathSum.put(source,1);

        // 前者记录满足到达距离的节点外观内容，后者记录到达节点和对应的节点外观
        HashSet<Integer> successnodes = new HashSet<>();
        Map<Integer,Set<Integer>> successResults = new HashMap<>();

        // 按层数进行BFS
        while(!currentAppearance.isEmpty()){
            int size = currentAppearance.size();
                // 计算并记录已访问时间
                int thisId = currentAppearance.poll();
                int thisNode = lifespans.get(thisId);
                ArrayList<Integer> thisTime = currentTime.poll();

                if(!this.neighbors.containsKey(thisNode)){
                    continue;
                }

                for(int neighbor:this.neighbors.get(thisNode)){
                    ArrayList<Integer> newTime = new ArrayList<>(thisTime);
                    newTime.retainAll(edgeToTimestamps.get(new Edge(thisNode,neighbor)));
                    // 剪枝策略1
                    if(newTime.isEmpty()){
                        continue;
                    }

                    // 第一次拜访neighbor，表示一定是最短路径
                    if(!visited.containsKey(neighbor)){
                        // 标记已访问，记录currLevel；
                        visited.put(neighbor,newTime);

                        // 在这里初始化以避免前方操作
                        sourceArray = new int[stampSum];
                        Arrays.fill(sourceArray,-1);
                        dist.put(neighbor,sourceArray);

                        // 更新最短距离的位图
                        for(int stamp:newTime){
                            dist.get(neighbor)[stamp] = dist.get(thisNode)[stamp]+1;
                        }

                        // 创建新的节点外观；
                        // 更新successor；
                        int neighborId = ++num;
                        lifespans.put(neighborId,neighbor);
                        lifespansStamp.put(neighborId,newTime);
                        nodeAppearance.computeIfAbsent(neighbor,k->new ArrayList<>()).add(neighborId);
                        predecessor.put(neighborId,thisId);

                        // 更新pathnum；
                        int timeSize = newTime.size();
                        int p = pathnum.getOrDefault(neighborId,0);
                        int q = pathSum.getOrDefault(neighbor,0);
                        pathnum.put(neighborId,pathnum.getOrDefault(neighborId,0)+timeSize);
                        pathSum.put(neighbor,pathSum.getOrDefault(neighbor,0)+timeSize);
                        increaseNode.add(neighborId);

                        // 更新BFS待遍历队列
                        currentAppearance.add(neighborId);
                        currentTime.add(newTime);

                        if (level>=successScale){
                            successnodes.add(neighborId);
                        }
                    } else{//标记
                        // 非第一次拜访neighbor，分两种情况
                        // 1.在新的时间上第一次拜访neighbor，更新这个时间的前身集合和最短距离
                        // 2.在已访问时间内再次拜访neighbor，如果是发现了新的路径和前身，则更新前身集合
                        // 不可能发现更短的路径，因为BFS分层执行.
                        // 取A-A∩B，记为restTime
                        ArrayList<Integer> restTime = new ArrayList<>(newTime);
                        restTime.removeAll(visited.get(neighbor));
                        // single和Shortest分别记录：情况23的结果更新区间树，情况3的结果更新前驱和pathnum
                        TreeSet<Integer> single = new TreeSet<>();
                        // restTime不为空，表示有其他时间也第一次访问到了neighbor，先处理未重复的这一部分
                        if(!restTime.isEmpty()){
                            single.addAll(restTime);
                            // 第一次访问更新最短距离位图即可
                            for(int stamp:restTime){
                                dist.get(neighbor)[stamp] = dist.get(thisNode)[stamp]+1;
                            }
                        }
                        // 已访问的时间中再次拜访neighbor，只判断是否有新的路径
                        ArrayList<Integer> alreadyTime = new ArrayList<>(newTime);
                        alreadyTime.retainAll(visited.get(neighbor));
                        if (!alreadyTime.isEmpty()){
                            // 转化成single依次处理
                            for(int stamp:alreadyTime){
                                // 判断走当前路径的距离【thisnode+1】是否等于先前记录的距离【neighbor】
                                // 只可能是大于等于，等于时添加路径
                                if(dist.get(neighbor)[stamp]==dist.get(thisNode)[stamp]+1){
//                                    System.out.println("情况3：存在多条路径");
                                    single.add(stamp);
                                }
                            }

                        }

                        // 满足情况2或3时，才更新区间树
                        if(!single.isEmpty()) {
                            // 合并满足两种情况的lifespan，位图不需要更新
                            // 添加区间树节点，新的区间树节点需要合并情况二三的lifespan
                            int timeSize = single.size();
                            ArrayList<Integer> resultTime = new ArrayList<>(single);
                            // 创建新的spanAppearance；
                            // 更新successor；
                            // 检查并初始化中间层的 Map
                            int neighborId = ++num;
                            lifespans.put(neighborId,neighbor);
                            lifespansStamp.put(neighborId,resultTime);
                            nodeAppearance.computeIfAbsent(neighbor,k->new ArrayList<>()).add(neighborId);
                            predecessor.put(neighborId,thisId);
                            // 更新pathnum；
                            pathnum.put(neighborId,pathnum.getOrDefault(neighborId,0)+timeSize);
                            pathSum.put(neighbor,pathSum.getOrDefault(neighbor,0)+timeSize);
                            int p = pathnum.getOrDefault(neighborId,0);
                            int q = pathSum.getOrDefault(neighbor,0);
//                            pathnum.put(neighborId,pathnum.getOrDefault(neighborId,0)+timeSize*pathnum.get(thisId));
//                            pathSum.put(neighbor,pathSum.getOrDefault(neighbor,0)+timeSize*pathnum.get(thisId));
//                            nodePathnum.put(neighbor,nodePathnum.get(neighbor)+timeSize*pathnum.get(thisId));
                            visited.compute(neighbor, (k, v) -> Stream.concat(v.stream(), resultTime.stream())
                                    .distinct()
                                    .collect(Collectors.toCollection(ArrayList::new)));
                            increaseNode.add(neighborId);
                            // 更新BFS待遍历队列
                            currentAppearance.add(neighborId);
                            currentTime.add(resultTime);
                            if (level>=successScale){
                                successnodes.add(neighborId);
                            }
                        }
                    }
                }
            level++;
            // 要更新层结构
        }

        if(level<=1){
//            System.out.println("无法扩张");
            long currentEndTime = System.currentTimeMillis();
            this.samplingTime+=currentEndTime-currentStartTime;
            return false;
        }
//        System.out.println("遍历步数为："+level);
        if(level>=this.maxDistance){
            this.maxDistance=level;
        }
        while(!increaseNode.isEmpty()){
            int thisAppearance = increaseNode.pop();
            int thisNode = lifespans.get(thisAppearance);
            if(successnodes.contains(thisAppearance)){
                if (!predecessor.values().contains(thisAppearance)){
                    successResults.computeIfAbsent(thisNode,k->new HashSet<>()).add(thisAppearance);
                }
            }
        }
        // 对于每个成功节点
        for (int succNode:successResults.keySet()){
            int sum = pathSum.get(succNode);
            Queue<Integer> currentReverse = new LinkedList<>();
            // 记录节点依赖度
            Map<Integer,Double> delta = new HashMap<>();
            // 对于每个成功节点的 appearance
            for(int succAppearance:successResults.get(succNode)){
                int sumApp = pathnum.get(succAppearance);
                currentReverse.add(succAppearance);
                while(!currentReverse.isEmpty()){
                    int thisAppearance = currentReverse.poll();
                    int thisNode = lifespans.get(thisAppearance);
                    int pre=predecessor.get(thisAppearance);
                    int preNode = lifespans.get(pre);
                    HashSet<Integer> thisAppearanceSuccessors = new HashSet<>();
                    thisAppearanceSuccessors.addAll(successResults.get(succNode));
                    delta.compute(preNode, (key, value) -> {
                        if (value == null) {
                            value = 0.0;
                        }
                        return value + (double)pathnum.get(thisAppearance)/pathSum.get(thisNode);
                    });
                }
            }
            for(Map.Entry<Integer, Double> entry:delta.entrySet()){
                if(entry.getKey()!=source){
                    this.Vw.get(entry.getKey()).add(entry.getValue());
                }
            }
        }

        if(!successResults.isEmpty()){
            System.out.println("成功找到"+successResults.size()+"个节点对，已有"+successSample+"个节点对");
            successSample+=successResults.values().stream()
                    .mapToInt(Set::size)
                    .sum();
        }
        return true;
    }
    // 近似计算TBC，输入accuracy parameter epsilon, confidence parameter delta.
    public Map<Integer,Double> CalculateTBC(double epsilon, double delta, int sampleSize){
        // 时间块去重采样
        Map<Integer, boolean[]> usedIntervalsMap = new HashMap<>();
        // 采样第i代；
        int i = 1;
//        int S1 = calculateS1(epsilon,delta);
//        int S1 = nodesArray.size()/5; // 节点对采样的采样次数：V/5
        int samplesum = 0;
        int successsum = 0;
        int falsesum = 0;
        int T = timestamps.size();
        // 采样为T的几分之一；
        int sampleInterval = 1;
        int subIntervalLength = T / sampleInterval;
//        Set<String> alreadyNode = new HashSet<>();
        Random random = new Random();
//        Set<Integer> printedPercentages = new HashSet<>();

        System.out.println("采样次数： "+sampleSize);

        // 随机
        int size = nodesArray.size();
        ArrayList<Integer> sampleSet = new ArrayList<>(nodesArray);
        // 度加权
//        int size = leftNodes.size();
//        ArrayList<Integer> sampleSet = new ArrayList<>(leftNodes);

        long currentStartTime = System.currentTimeMillis();
        while(successSample<sampleSize){
            // 时间块去重采样
            int index=-1;
            boolean[] usedIntervals;
            int x =-1;

            // 节点度加权去重采样
            do{
                x = randomWeightedSample(random);
                usedIntervals = usedIntervalsMap.computeIfAbsent(x, k -> new boolean[sampleInterval]);
                index = findFirstFalseIndex(usedIntervals);
                 if(index!=-1){
                     usedIntervals[index] = true;
                }
                if(allTrue(usedIntervals)){
                    this.deletedNodes.add(x);
                    this.degreeWeights.remove(x);
                    updateWeight(degreeWeights);
                }
                if(x==-1){
                    System.out.println("可用节点用光啦");
                    return null;
                }
            }while(index == -1);
            int start = index * subIntervalLength;
            int end = start + subIntervalLength - 1;

            if(CalculateSSSPModify(x,start,end)){
                successsum++;
            }else{
                falsesum++;
            }
            samplesum++;
        }

        System.out.println("—————————————— results ————————————————");
        System.out.println("最远距离："+maxDistance);
        System.out.println("采样效率："+(double)successSample/successsum+"，采样成功次数："+successsum+"，采样节点对数："+successSample);

        long currentSampleTime = System.currentTimeMillis();

        System.out.printf("采样时长：%d 毫秒.\n",(this.samplingTime));
        System.out.printf("计算时长：%d 毫秒.\n",(this.calculateTime));
        System.out.printf("总时长：%d 毫秒.\n",(currentSampleTime-currentStartTime));

        int Si = successSample;
        double w = Simulated_Annealing(this.Vw,Si);
        double eta = delta/ (int) Math.pow(2, i);
        double alpha = calculateAlpha(eta,Si,w);
        double ub = calculateUb(w,eta,Si,alpha);

        System.out.println("采样数量为："+successSample+" 时，采样下界是:"+ub);

        long currentEndTime = System.currentTimeMillis();
        System.out.printf("退火时长：%d 毫秒.\n",(currentEndTime-currentSampleTime));
        System.out.printf("执行时长：%d 毫秒.\n",(currentEndTime-currentStartTime));

        // 展示结果
        Map<Integer,Double> betweenness = new HashMap<>();
        for (Map.Entry<Integer, List<Double>> entry : Vw.entrySet()) {
            List<Double> values = entry.getValue();
            double sum = 0;
            // 计算平均值
            for (Double value : values) {
                sum += value;
            }
            double average = sum;
            // 除以10000并存入新Map
            betweenness.put(entry.getKey(), average/Si);
        }
        return betweenness;
    }

    public static boolean allTrue(boolean[] v) {
        for (boolean x : v) {
            if (!x)
                return false;
        }
        return true;
    }

    public static double Simulated_Annealing( Map<Integer, List<Double>> vs,int Si) {
        // 假设这里添加了向量到vs中
        Random random = new Random();
        double initialC = 1;
        double c = initialC;
        double temperature = 100;
        double coolingRate = 0.01;
        double minOmega = omegaStar(vs, c, Si);
        double optimalC = c;
        while (temperature > 1) {
            double newC = c + (random.nextDouble() - 0.5) * temperature;
            // 确保是正实数
            if(newC>0){
                double newOmega = omegaStar(vs, newC,Si);
                if (newOmega < minOmega || Math.exp((minOmega - newOmega) / temperature) > random.nextDouble()) {
                    c = newC;
                    minOmega = newOmega;
                    optimalC = c;
                }
            }
            temperature *= 1 - coolingRate;
        }
        System.out.println("Optimal c: " + optimalC);
        System.out.println("Omega*: " + minOmega);
        return minOmega;
    }

    public static double calculateAlpha(double eta, double S, double omegaStar) {
        double numerator = Math.log(2 / eta);
        double denominator = Math.log(2 / eta) + Math.sqrt((2 * S * omegaStar + Math.log(2 / eta)) * Math.log(2 / eta));
        return numerator / denominator;
    }

    public static double calculateUb(double omegaStar, double eta, double S, double alpha) {
        double firstPart = omegaStar / (1 - alpha);
        double secondPart = Math.log(2 / eta) / (2 * S * alpha * (1 - alpha));
        double thirdPart = Math.sqrt(Math.log(2 / eta) / (2 * S));
        return firstPart + secondPart + thirdPart;
    }
    public static double norm(List<Double> v) {
        double sum = 0;
        for (double d : v) {
            sum += d * d;
        }
        return Math.sqrt(sum);
    }

    public static double omegaStar(Map<Integer, List<Double>> vs, double c, int sSize) {
        double sumExp = 0;
        for (List<Double> v : vs.values()) {
            double norm = norm(v);
            double expTerm = Math.exp(c * c * norm * norm / (2 * sSize * sSize));
            sumExp += expTerm;
        }
        return (1 / c) * Math.log(sumExp);
    }

    public int[] randomSample(Random random){

        ArrayList<Integer> selectedIntegers = new ArrayList<>();

        // 随机选择第一个元素
        int index = random.nextInt(nodes.size());
        int first = nodesArray.get(index);
//        int secondIndex;
//        int second;
//        do {
//            secondIndex = random.nextInt(nodes.size());
//            second = nodesArray.get(secondIndex);
//        } while (second == first);

        int[] result = new int[2];
        result[0] = first;
//        result[1] = second;
        return result;
    }

    public int randomWeightedSample(Random random) {
        // 随机选择第一个元素
        int first = selectNodeByWeightInterval(sssrandom, this.cumulativeWeights, this.totalWeight);

        // 随机选择第二个不同的元素
//        int second = -1;
//        do {
//            second = selectNodeByWeight(sssrandom, this.cumulativeWeights, this.totalWeight);
//        } while (second == first);
//
//        int[] result = new int[2];
//        result[0] = first;
//        result[1] = second;
        return first;
    }

    public int selectNodeByWeight(Random random, TreeMap<Integer, Integer> cumulativeWeights, int totalWeight) {
        int randomWeight = random.nextInt(totalWeight) + 1; // 生成 [1, totalWeight] 之间的随机数
        for (Map.Entry<Integer, Integer> entry : cumulativeWeights.entrySet()) {
            if (randomWeight <= entry.getValue()) {
                return entry.getKey();
            }
        }
        return -1;
    }

    public int selectNodeByWeightInterval(Random random, TreeMap<Integer, Integer> cumulativeWeights, int totalWeight) {
        int maxRetry = 100;
        for (int i = 0; i < maxRetry; i++) {
            int randomWeight = sssrandom.nextInt(totalWeight) + 1;
            Integer floorKey = cumulativeWeights.floorKey(randomWeight);
            if (floorKey == null) {
                continue;
            }
            int candidate = cumulativeWeights.get(floorKey);
            if (deletedNodes.contains(candidate)) {
                // 如果节点已被删除，重试（重新生成随机数）
                continue;
            }
            return candidate;
        }
        // 重试maxRetry次后仍未找到，返回-1
        return -1;
    }

    public void updateWeight(Map<Integer, Integer> importanceMap){
        this.cumulativeWeights.clear();
        // 生成累积权重
        int cumulativeWeight = 0;
        for (int node : importanceMap.keySet()) {
            cumulativeWeight += importanceMap.getOrDefault(node, 0);
//            this.cumulativeWeights.put(node,cumulativeWeight); // 累积权重的普通法
            this.cumulativeWeights.put(cumulativeWeight,node); // 累积权重的区间去重法
        }
        this.totalWeight = cumulativeWeight;
    }

    public int findFirstFalseIndex(boolean[] array) {
        for (int i = 0; i < array.length; i++) {
            if (!array[i]) {
                return i;
            }
        }
        return -1;
    }
}
