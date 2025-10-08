package org.example;

import java.io.*;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class HBCReadData {
    public static int plusnum = 0;
    // 存储数据的map <（u，v)，时间戳集合）>
    public Map<Edge, Set<Integer>> edgeToTimestamps=new HashMap<>();
    // 边数量
    public int edgeNum;
    // 存储点的邻居节点
    public Map<Integer, HashSet<Integer>> neighbors=new HashMap<Integer,HashSet<Integer>>();
    // 存在的snapshot
    public HashSet<Integer> timestamps = new HashSet<>();
    // 存储节点集
    public HashSet<Integer> nodes = new HashSet<>();
    // 存储节点集
    public ArrayList<Integer> nodesArray = new ArrayList<>();
    // 临时保存结果
    public Map<Integer, Double> resultS  = new HashMap<>();

    public HBCReadData(String fileName) {
        BufferedReader reader = null;
        int alreadyLines = 0;
        int alreadyNum = 0;
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
                    System.out.println("有独立环");
                    continue;
                }
                edgeToTimestamps.compute(new Edge(start, end), (edge, list) -> {
                    if (list == null) {
                        list = new HashSet<>();
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
            edgeNum = alreadyLines+alreadyNum*1000000;
            System.out.println("数据集大小："+fileName+" ,节点数量：" +nodesArray.size()+" ,边数量： "+edgeNum+" ,snapshots数量： "+timestamps.size());
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

    // 正统单源最短路径，时间戳离散存储
    public Map<Integer, Double> CalculateSSSP(int source, int interval_s, int interval_t){
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
        // 记录节点路径数量
        Map<Integer,Integer> pathSum = new HashMap<>();

        // 记录节点依赖度
        Map<Integer,Double> delta = new HashMap<>();
        // 记录拜访记录
        Map<Integer,ArrayList<Integer>> visited = new HashMap<>();
        Map<Integer,ArrayList<Integer>> beforeLevelVisited = new HashMap<>();
        // 介数结果
        Map<Integer,Double> betweenness = new HashMap<>();

        // 初始化source访问距离
        int[] sourceArray = new int[stampSum];
        Arrays.fill(sourceArray,0);
        dist.put(source,sourceArray);
        // 初始化source访问时间
        ArrayList<Integer> initial = new ArrayList<>();
        for (int i = 0; i <stampSum+1; i++) {
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

        // lifespan不带重复时间戳
        // 按层数进行BFS
        while(!currentAppearance.isEmpty()){
            // 计算并记录已访问时间
            int thisId = currentAppearance.poll();
            int thisNode = lifespans.get(thisId);
            increaseNode.add(thisId);
            ArrayList<Integer> thisTime = currentTime.poll();

            if(!this.neighbors.containsKey(thisNode)){
                continue;
            }
//                visited.put(thisNode,thisTime);
            // 标记
            for(int neighbor:this.neighbors.get(thisNode)){
                // BFS遍历thisNode-->neighbor这条边，得到此时的lifespan，即newTime
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
                    pathnum.put(neighborId,pathnum.getOrDefault(neighborId,0)+timeSize);
                    pathSum.put(neighbor,pathSum.getOrDefault(neighbor,0)+timeSize);

                    // 更新BFS待遍历队列
                    currentAppearance.add(neighborId);
                    currentTime.add(newTime);

                } else {
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

                        // 更新BFS待遍历队列
                        currentAppearance.add(neighborId);
                        currentTime.add(resultTime);
                        visited.compute(neighbor, (k, v) -> Stream.concat(v.stream(), resultTime.stream())
                                .distinct()
                                .collect(Collectors.toCollection(ArrayList::new)));
                    }
                }
            }

            level++;
            // 要更新层结构
        }

        // 长度递减地弹出节点外观
        HashSet< Integer> aaa = new HashSet<>();
        while(!increaseNode.isEmpty()){
            int thisAppearance = increaseNode.pop();
            int thisNode = lifespans.get(thisAppearance);


            Integer pre = predecessor.get(thisAppearance);
            if (pre != null) {
                int preNode = lifespans.get(pre);
                delta.compute(preNode, (key, value) -> {
                        return value + (double) pathnum.get(thisAppearance) / pathSum.get(thisNode);
                    });
                    // delta.get(preNode)+=(double)pathnum.get(pre)/pathnum.get(thisAppearance)*(1+delta.get(thisNode));

                if(thisNode!=source){
                    betweenness.put(thisNode,betweenness.getOrDefault(thisNode,0.0)+delta.getOrDefault(thisNode,0.0));
                }
            }
        }
        return betweenness;
    }

    // HBC计算
    public Map<Integer,Double> CalculateHBC(){
        long currentStartTime = System.currentTimeMillis();

        int num =0;
        int step = 10;
        Map<Integer,Double> betweenness = new HashMap<>();
        int nodeNum = nodesArray.size();
        for(int node:nodesArray){
            Map<Integer,Double> itemResult = CalculateSSSP(node,0,10000);
            for(Map.Entry<Integer, Double> entry : itemResult.entrySet()){
                int nodeKey = entry.getKey();
                betweenness.put(nodeKey,betweenness.getOrDefault(nodeKey,0.0)+entry.getValue());
            }
        }

        long currentEndTime = System.currentTimeMillis();
        System.out.printf("执行时长：%d 毫秒.\n",(currentEndTime-currentStartTime));

        betweenness.forEach((key, value) ->
                resultS.put(key, value / nodeNum / (nodeNum + 1)));

        return betweenness;
    }
}
