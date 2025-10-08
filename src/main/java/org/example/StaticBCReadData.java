package org.example;

import java.io.*;
import java.util.*;
import java.util.stream.Collectors;

public class StaticBCReadData {
    // 边数量
    public int edgesum;
    // 存储边集，外层Map的key是时间戳，value是该时间戳下的snapshot边集
    public Map<Integer, Map<Integer, Set<Integer>>> edges = new HashMap<>();
    // 存储节点集
    public Set<Integer> nodes = new HashSet<>();
    // merged graph edges
    public Map<Integer, Set<Integer>> mergedEdges = new HashMap<>();
    // 记录每个节点在每个snapshot的介数中心性前20名的次数
    public Map<Integer, Integer> bcTop20Count = new HashMap<>();
    // 记录每个节点的颜色深度
    public Map<Integer, Double> colorDepth = new HashMap<>();
    // 存储每个节点染色label
    Map<Integer,Integer> nodeLabelMap = new HashMap<>();
    public static void main(String[] args) {
        long currentStartTime = System.currentTimeMillis();
        String filePath = "D:/code/superuser.txt";
        StaticBCReadData sbc = new StaticBCReadData(filePath);
        long currentComputeTime = System.currentTimeMillis();
        sbc.calculateBetweennessRank(50);
        long currentEndTime = System.currentTimeMillis();
        // infectious-hour-21.602s
        // infectious-hour-21.628s
        System.out.printf("执行时长：%d 毫秒.\n",(currentEndTime-currentComputeTime));
    }
    public StaticBCReadData(String fileName) {
        BufferedReader reader = null;
        try {
            // 初始化BufferedReader以读取文件
            reader = new BufferedReader(new FileReader(fileName));
            String line;

            // 循环读取文件的每一行
            while ((line = reader.readLine()) != null) {
                // 通过空格分隔每行数据为多个部分
                String[] tokens = line.split("\\s+");
                // 将第一部分转换为整型的时间戳
                int timestamp = Integer.parseInt(tokens[0]);
                // 获取开始和结束标志
                int start = Integer.parseInt(tokens[1]);
                int end = Integer.parseInt(tokens[2]);

                // 更新节点集
                nodes.add(start);
                nodes.add(end);

                // 更新边集
                if(edges.computeIfAbsent(timestamp, k -> new HashMap<>())
                        .computeIfAbsent(start, k -> new HashSet<>())
                        .add(end)){
                    edgesum++;
                }

                mergedEdges.compute(start, (node, set) -> {
                    if (set == null) {
                        set = new HashSet<>();
                    }
                    set.add(end);
                    return set;
                });
            }

            for(int node:nodes){
                bcTop20Count.put(node,0);
            }

            System.out.println("已导入边数: " + edgesum);
        } catch (FileNotFoundException e) {
            // 如果文件未找到，抛出运行时异常
            throw new RuntimeException(e);
        } catch (IOException e) {
            // 如果发生IO异常，抛出运行时异常
            throw new RuntimeException(e);
        } finally {
            // 确保BufferedReader在使用后被正确关闭
            if (reader != null) {
                try {
                    reader.close();
                } catch (IOException e) {
                    // 在关闭过程中发生的IO异常同样抛出运行时异常
                    throw new RuntimeException(e);
                }
            }
        }
    }

    // 计算每个时间戳下的snapshot的每个节点的介数中心性
    public Map<Integer, Integer> calculateBetweennessRank(int topTimes) {
        int sum =0;
        System.out.println("快照数量："+edges.keySet().size());

        Map<Integer, Map<Integer,Integer>> bcTop = new HashMap<>();

        bcTop.put(topTimes,new HashMap<>());


        for (Map.Entry<Integer, Map<Integer, Set<Integer>>> entry : edges.entrySet()) {
            Map<Integer, Set<Integer>> snapshotEdges = entry.getValue();

            // 构建当前时间戳下的图
            Graph graph = new Graph();
            for (Map.Entry<Integer, Set<Integer>> nodeEntry : snapshotEdges.entrySet()) {
                int source = nodeEntry.getKey();
                for (int target : nodeEntry.getValue()) {
                    graph.addEdge(source, target);
                }

            }

            // 计算介数中心性
            Map<Integer, Double> betweennessCentrality = graph.calculateBetweennessCentrality();
            System.out.println("当前时间戳已完成：" + entry.getKey());
            sum+=graph.pathsum;
            // 获取介数中心性前x名的节点
            List<Map.Entry<Integer, Double>> top20 = betweennessCentrality.entrySet()
                    .stream()
                    .sorted((e1, e2) -> e2.getValue().compareTo(e1.getValue()))
                    .limit(topTimes)
                    .collect(Collectors.toList());

            // 记录每个节点在前x名的次数
            for (Map.Entry<Integer, Double> nodeEntry : top20) {
                int nodeId = nodeEntry.getKey();
                bcTop.get(topTimes).put(nodeId,bcTop.get(topTimes).getOrDefault(nodeId,0)+1);
//                    bcTop20Count.put(nodeId, bcTop20Count.getOrDefault(nodeId, 0) + 1);
            }

        }
        System.out.println("pathsum:"+sum);
        List<Map.Entry<Integer, Integer>> top100 = bcTop20Count.entrySet()
                .stream()
                .sorted(Map.Entry.<Integer, Integer>comparingByValue().reversed())
                .limit(100)
                .collect(Collectors.toList());
        // 输出结果
//        top100.forEach(entry -> System.out.println("Key: " + entry.getKey() + ", Value: " + entry.getValue()));

        return bcTop.get(topTimes);
    }

    class Graph {
        private Map<Integer, List<Integer>> adjacencyList = new HashMap<>();
        private Set<Integer> allNodes = new HashSet<>();
        public int pathsum = 0;
        public void addEdge(int source, int target) {
            adjacencyList.computeIfAbsent(source, k -> new ArrayList<>()).add(target);
            allNodes.add(source);
            allNodes.add(target);
        }

        public Map<Integer, Double> calculateBetweennessCentrality() {
            Map<Integer, Double> betweennessCentrality = new HashMap<>();
            for (int node : allNodes){
                betweennessCentrality.put(node, 0.0);
            }

            for (int s: allNodes) {
                Deque<Integer> stack = new ArrayDeque<>();
                Map<Integer, Integer> sigma = new HashMap<>();
                Map<Integer, Integer> d = new HashMap<>();
                Map<Integer, List<Integer>> P = new HashMap<>();
                Map<Integer, Double> delta = new HashMap<>();

                for (int v : allNodes) {
                    sigma.put(v, 0);
                    d.put(v, -1);
                    P.put(v, new ArrayList<>());
                    delta.put(v, 0.0);
                }

                sigma.put(s, 1);
                d.put(s, 0);

                Queue<Integer> queue = new LinkedList<>();
                queue.add(s);

                while (!queue.isEmpty()) {
                    int v = queue.poll();
                    stack.push(v);
                    for (int w : adjacencyList.getOrDefault(v, Collections.emptyList())) {
                        if (d.get(w) < 0) {
                            queue.add(w);
                            d.put(w, d.get(v) + 1);
                        }
                        if (d.get(w) == d.get(v) + 1) {
                            sigma.put(w, sigma.get(w) + sigma.get(v));
                            P.get(w).add(v);
                            this.pathsum++;
                        }
                    }
                }

                while (!stack.isEmpty()) {
                    int w = stack.pop();
                    for (int v : P.get(w)) {
                        delta.put(v, delta.get(v) + (sigma.get(v) / (double) sigma.get(w)) * (1 + delta.get(w)));
                    }

                    betweennessCentrality.put(w, betweennessCentrality.get(w) + delta.get(w));

                }
            }

            // 归一化介数中心性
            int n = allNodes.size();
            for (int node : allNodes) {
                if (n > 2) {
                    betweennessCentrality.put(node, betweennessCentrality.get(node) / ((n - 1) * (n - 2)));
                } else {
                    betweennessCentrality.put(node, 0.0);
                }
            }

            return betweennessCentrality;
        }
    }
}



