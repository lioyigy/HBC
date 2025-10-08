package org.example;

import org.checkerframework.checker.units.qual.A;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.sql.Array;
import java.sql.SQLOutput;
import java.util.*;
import java.util.concurrent.*;
import java.util.stream.Collectors;
//import com.google.common.base.SizeEstimator;

public class Main {
    public static void main(String[] args) {

        // 读取文件数据
        String filePath = "D:/code/emailKonect.txt";
        int lastSeparatorIndex = filePath.lastIndexOf('/');
        HBCReadData hbc = new HBCReadData(filePath);
        // 执行方法并计时
        long currentStartTime = System.currentTimeMillis();
        // 执行计算
        List<Integer> sortedKeys1  =  hbc.CalculateHBC().entrySet().stream()
                .sorted(Map.Entry.<Integer, Double>comparingByValue().reversed())
                .map(Map.Entry::getKey)
                .collect(Collectors.toList());

        long currentEndTime = System.currentTimeMillis();
        System.out.printf("历史执行时长：%d 毫秒.\n",(currentEndTime-currentStartTime));

//        // 执行方法并计时
        currentStartTime = System.currentTimeMillis();
        // 执行计算
        StaticBCReadData sbc = new StaticBCReadData(filePath);
        // 传入的是被记录进统计rank的rank名次
        Map<Integer,Integer> s = sbc.calculateBetweennessRank(hbc.nodesArray.size()*5/100);
//        Map<Integer,Integer> s = sbc.calculateBetweennessRank(10 );
        Map<Integer, Double> newMap = new HashMap<>();
        for (Map.Entry<Integer, Integer> entry : s.entrySet()) {
            newMap.put(entry.getKey(), entry.getValue().doubleValue());
        }
        List<Integer> sortedKeys2 = newMap.entrySet().stream()
                .sorted(Map.Entry.<Integer, Double>comparingByValue().reversed())
                .map(Map.Entry::getKey)
                .collect(Collectors.toList());
        currentEndTime = System.currentTimeMillis();
        System.out.printf("静态执行时长：%d 毫秒.\n",(currentEndTime-currentStartTime));
    }

}