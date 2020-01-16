#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <random>
#include <string>
#include <iterator>
#include <cmath>
#include <bits/stdc++.h>
#include <bitset>

#include "structs.h"
#include "funct.h"
#include "hash.h"
#include "cluster_funct.h"

using namespace std;

//epistrefei ena vector item apo th grammh tou arxeiou pou pairnei san orisma
Vector_Item initialize_item(string line){
  Vector_Item item;
  int id_flag = 1;
  string digits = "";
  //pairnw ena ena ta xi
  for (auto x : line){
    if (x == ','){
      if(id_flag == 1){
        item.set_id(digits);
        id_flag = 0;
      }
      else{
        item.push(stod(digits));
      }
      digits = "";
    }
    else{
      digits = digits + x;
    }
  }
  item.push(stod(digits));
  return item;
}

//arxikopoiei ton pinaka me ta data apo to input file
int Initialize_Dataset_Vector(string filename, vector<Vector_Item>* Items){
    string digits = "";
    string line;
    int id_flag = 1;
    int c = 0;
    fstream file;

    file.open(filename);
    // extracting from the file
    //while (file.good())
    for(int i=0; i<1000; i++)
    {
        if (!getline (file, line)) break;

        Vector_Item item = initialize_item(line);

        //apo8hkeuw to dataset se ena vector
        Items->push_back(item);
        id_flag = 1;
        c++;
    }
    file.close();

    return c;
}

void configuration(string conf_file, int* numof_clusters, int* numof_grids, int* numofV_hashtables, int* numofV_hashfuncts){
    ifstream Configfile;
    Configfile.open(conf_file, ios::in);

    string line;
    int flag=0;
    int var_counter=1;

    while(Configfile.good()){
        if (!getline (Configfile, line)) break;
        string var="";

        for(auto x: line){
            if(flag==1){
                var=var+x;
                if(var_counter==1){
                    *numof_clusters=stoi(var);
                }else if(var_counter==2){
                    *numof_grids=stoi(var);
                }else if(var_counter==3){
                    *numofV_hashtables=stoi(var);
                }else if(var_counter==4){
                    *numofV_hashfuncts=stoi(var);
                }
            }
            if(x==' ')  flag=1;
        }
        var_counter++;
        var="";
        flag=0;
    }

    return;
}

Vector_Item ExactNN(vector<Vector_Item> Items, Vector_Item item, int c, int* ExactNN_dist){
    int NN_pos = -1;
    int d = item.get_vector().size();
    int NN_dist;
    long int min = 10000000000000;
    int i;

    for(i=0; i<c; i++){

        Vector_Item neighbor_item = Items.at(i);
        NN_dist=distance_l1(item.get_vector(), neighbor_item.get_vector(), d);

        if(NN_dist<min){
            NN_pos= i;
            min=NN_dist;
        }
    }

    *ExactNN_dist = min;
    return Items.at(NN_pos);
}

Vector_Item AproximateNN(vector<Vector_Item> Items, Vector_Item item, vector<Bucket>** HT, int buckets, int k, int L, long int m, int M, int W, int* AprNN_dist){
    int NN_position = -1;
    long int NN_dist = 10000000000000;

    int d = item.get_vector().size();

    for(int l=0; l<L; l++){
        //ftiaxnw k hash sunarthseis
        int key = hash_key(item, buckets, d, k, W, M, m);//g%buckets;
        //auto found = HT.at(l).find(key);
        if((key<0) || (key>=buckets)) key = 1;

        if(HT[l]->at(key).get_point_pos().size() != 0){//(found != HT.at(l).end()){
            //auto itr = HT.at(l).equal_range(key);

            //gia ka8e stoixeio me kleidei key
            int retrieved_items = 0;
            for(int p=0; p<HT[l]->at(key).get_point_pos().size(); p++){//(auto it = itr.first; it != itr.second; it++) {
                retrieved_items++;
                int pos = HT[l]->at(key).get_point_pos().at(p);
                Vector_Item neighbor_item = Items.at(pos);//Items.at(it->second); //edw 8a xreiastei mia find an to id den einai 0-n

                if(retrieved_items > 3*L) break;

                if(distance_l1(item.get_vector(), neighbor_item.get_vector(), d) < NN_dist){
                   NN_position = pos;//it->second;
                   NN_dist = distance_l1(item.get_vector(), neighbor_item.get_vector(), d);
                }
            }
        }
        else
            cout << "!!!!!!!!!!!there is no key " << key << endl;
    }
    *AprNN_dist = NN_dist;

    return Items.at(NN_position);
}

void write_results(string outfile, int I, int A, int U, int k, vector<Cluster> clusters, vector<Vector_Item> centroids, double cltime, int complete, vector<double> s, double stotal){

    ofstream Outfile;
    Outfile.open(outfile, ios::trunc);

    string line;

    Outfile << "Algorithm: I" << I << "A" << A << "U" << U << endl;

    for(int i=0;i<k;i++){
        if(clusters.at(i).get_center_id() == "new_c"){
            Outfile << "CLUSTER-" << i+1 << " {size: " << clusters.at(i).get_positions().size() << ", centroid: ( ";
            for(int j=0; j<centroids.at(i).get_vector().size(); j++)
                Outfile << centroids.at(i).get_vector().at(j) << " ";
            Outfile << ") }" << endl;
        }
        else
            Outfile << "CLUSTER-" << i+1 << " {size: " << clusters.at(i).get_positions().size() << ", centroid: " << clusters.at(i).get_center_id() << "}" << endl;
    }

    Outfile << "clustering_time: " << cltime << endl;

    Outfile << "Silhouette: [ ";
    for(int i=0; i<centroids.size(); i++){
        Outfile << s.at(i) << ", ";
    }
    Outfile << "stotal = " << stotal << " ]" << endl;

    if(complete){
        for(int i=0; i<clusters.size(); i++){
            Outfile << "Cluster-" << i+1 << " { ";
            for(int j=0; j<clusters.at(i).get_positions().size(); j++)
                Outfile << clusters.at(i).get_positions().at(j) << " ";
            Outfile << " }" << endl;
        }
    }

    Outfile << endl;
    Outfile.close();

    return;
}
