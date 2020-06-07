import csv
import operator
x=['Jan 31 00:09:39 ubuntu.local ticky: INFO Created ticket [#4217] (mdouglas)', 'Jan 31 00:16:25 ubuntu.local ticky: INFO Closed ticket [#1754] (noel)', 'Jan 31 00:21:30 ubuntu.local ticky: ERROR The ticket was modified while updating (breee)', 'Jan 31 00:44:34 ubuntu.local ticky: ERROR Permission denied while closing ticket (ac)', 'Jan 31 01:00:50 ubuntu.local ticky: INFO Commented on ticket [#4709] (blossom)', 'Jan 31 01:29:16 ubuntu.local ticky: INFO Commented on ticket [#6518] (rr.robinson)', 'Jan 31 01:33:12 ubuntu.local ticky: ERROR Tried to add information to closed ticket (mcintosh)', 'Jan 31 01:43:10 ubuntu.local ticky: ERROR Tried to add information to closed ticket (jackowens)', 'Jan 31 01:49:29 ubuntu.local ticky: ERROR Tried to add information to closed ticket (mdouglas)', 'Jan 31 02:30:04 ubuntu.local ticky: ERROR Timeout while retrieving information (oren)', "Jan 31 02:55:31 ubuntu.local ticky: ERROR Ticket doesn't exist (xlg)", 'Jan 31 03:05:35 ubuntu.local ticky: ERROR Timeout while retrieving information (ahmed.miller)', "Jan 31 03:08:55 ubuntu.local ticky: ERROR Ticket doesn't exist (blossom)", 'Jan 31 03:39:27 ubuntu.local ticky: ERROR The ticket was modified while updating (bpacheco)', "Jan 31 03:47:24 ubuntu.local ticky: ERROR Ticket doesn't exist (enim.non)", 'Jan 31 04:30:04 ubuntu.local ticky: ERROR Permission denied while closing ticket (rr.robinson)', 'Jan 31 04:31:49 ubuntu.local ticky: ERROR Tried to add information to closed ticket (oren)', 'Jan 31 04:32:49 ubuntu.local ticky: ERROR Timeout while retrieving information (mcintosh)', 'Jan 31 04:44:23 ubuntu.local ticky: ERROR Timeout while retrieving information (ahmed.miller)', 'Jan 31 04:44:46 ubuntu.local ticky: ERROR Connection to DB failed (jackowens)', 'Jan 31 04:49:28 ubuntu.local ticky: ERROR Permission denied while closing ticket (flavia)', 'Jan 31 05:12:39 ubuntu.local ticky: ERROR Tried to add information to closed ticket (oren)', 'Jan 31 05:18:45 ubuntu.local ticky: ERROR Tried to add information to closed ticket (sri)', 'Jan 31 05:23:14 ubuntu.local ticky: INFO Commented on ticket [#1097] (breee)', 'Jan 31 05:35:00 ubuntu.local ticky: ERROR Connection to DB failed (nonummy)', 'Jan 31 05:45:30 ubuntu.local ticky: INFO Created ticket [#7115] (noel)', 'Jan 31 05:51:30 ubuntu.local ticky: ERROR The ticket was modified while updating (flavia)', 'Jan 31 05:57:46 ubuntu.local ticky: INFO Commented on ticket [#2253] (nonummy)', 'Jan 31 06:12:02 ubuntu.local ticky: ERROR Connection to DB failed (oren)', 'Jan 31 06:26:38 ubuntu.local ticky: ERROR Timeout while retrieving information (xlg)', 'Jan 31 06:32:26 ubuntu.local ticky: INFO Created ticket [#7298] (ahmed.miller)', 'Jan 31 06:36:25 ubuntu.local ticky: ERROR Timeout while retrieving information (flavia)', 'Jan 31 06:57:00 ubuntu.local ticky: ERROR Connection to DB failed (jackowens)', 'Jan 31 06:59:57 ubuntu.local ticky: INFO Commented on ticket [#7255] (oren)', "Jan 31 07:59:56 ubuntu.local ticky: ERROR Ticket doesn't exist (flavia)", 'Jan 31 08:01:40 ubuntu.local ticky: ERROR Tried to add information to closed ticket (jackowens)', 'Jan 31 08:03:19 ubuntu.local ticky: INFO Closed ticket [#1712] (britanni)', 'Jan 31 08:22:37 ubuntu.local ticky: INFO Created ticket [#2860] (mcintosh)', 'Jan 31 08:28:07 ubuntu.local ticky: ERROR Timeout while retrieving information (montanap)', 'Jan 31 08:49:15 ubuntu.local ticky: ERROR Permission denied while closing ticket (britanni)', 'Jan 31 08:50:50 ubuntu.local ticky: ERROR Permission denied while closing ticket (montanap)', 'Jan 31 09:04:27 ubuntu.local ticky: ERROR Tried to add information to closed ticket (noel)', 'Jan 31 09:15:41 ubuntu.local ticky: ERROR Timeout while retrieving information (oren)', 'Jan 31 09:18:47 ubuntu.local ticky: INFO Commented on ticket [#8385] (mdouglas)', 'Jan 31 09:28:18 ubuntu.local ticky: INFO Closed ticket [#2452] (jackowens)', 'Jan 31 09:41:16 ubuntu.local ticky: ERROR Connection to DB failed (ac)', 'Jan 31 10:11:35 ubuntu.local ticky: ERROR Timeout while retrieving information (blossom)', 'Jan 31 10:21:36 ubuntu.local ticky: ERROR Permission denied while closing ticket (montanap)', 'Jan 31 11:04:02 ubuntu.local ticky: ERROR Tried to add information to closed ticket (breee)', 'Jan 31 11:19:37 ubuntu.local ticky: ERROR Connection to DB failed (sri)', 'Jan 31 11:22:06 ubuntu.local ticky: ERROR Timeout while retrieving information (montanap)', 'Jan 31 11:31:34 ubuntu.local ticky: ERROR Permission denied while closing ticket (ahmed.miller)', 'Jan 31 11:40:25 ubuntu.local ticky: ERROR Connection to DB failed (mai.hendrix)', 'Jan 31 11:47:07 ubuntu.local ticky: INFO Commented on ticket [#4562] (ac)', 'Jan 31 11:58:33 ubuntu.local ticky: ERROR Tried to add information to closed ticket (ahmed.miller)', 'Jan 31 12:00:17 ubuntu.local ticky: INFO Created ticket [#7897] (kirknixon)', 'Jan 31 12:02:49 ubuntu.local ticky: ERROR Permission denied while closing ticket (mai.hendrix)', 'Jan 31 12:20:23 ubuntu.local ticky: ERROR Connection to DB failed (kirknixon)', "Jan 31 12:20:40 ubuntu.local ticky: ERROR Ticket doesn't exist (flavia)", 'Jan 31 12:24:32 ubuntu.local ticky: INFO Created ticket [#5784] (sri)', 'Jan 31 12:50:10 ubuntu.local ticky: ERROR Permission denied while closing ticket (blossom)', 'Jan 31 12:58:16 ubuntu.local ticky: ERROR Tried to add information to closed ticket (nonummy)', 'Jan 31 13:08:10 ubuntu.local ticky: INFO Closed ticket [#8685] (rr.robinson)', 'Jan 31 13:48:45 ubuntu.local ticky: ERROR The ticket was modified while updating (breee)', 'Jan 31 14:13:00 ubuntu.local ticky: INFO Commented on ticket [#4225] (noel)', 'Jan 31 14:38:50 ubuntu.local ticky: ERROR The ticket was modified while updating (enim.non)', 'Jan 31 14:41:18 ubuntu.local ticky: ERROR Timeout while retrieving information (xlg)', 'Jan 31 14:45:55 ubuntu.local ticky: INFO Closed ticket [#7948] (noel)', 'Jan 31 14:50:41 ubuntu.local ticky: INFO Commented on ticket [#8628] (noel)', 'Jan 31 14:56:35 ubuntu.local ticky: ERROR Tried to add information to closed ticket (noel)', "Jan 31 15:27:53 ubuntu.local ticky: ERROR Ticket doesn't exist (blossom)", 'Jan 31 15:28:15 ubuntu.local ticky: ERROR Permission denied while closing ticket (enim.non)', 'Jan 31 15:44:25 ubuntu.local ticky: INFO Closed ticket [#7333] (enim.non)', 'Jan 31 16:17:20 ubuntu.local ticky: INFO Commented on ticket [#1653] (noel)', 'Jan 31 16:19:40 ubuntu.local ticky: ERROR The ticket was modified while updating (mdouglas)', 'Jan 31 16:24:31 ubuntu.local ticky: INFO Created ticket [#5455] (ac)', 'Jan 31 16:35:46 ubuntu.local ticky: ERROR Timeout while retrieving information (oren)', 'Jan 31 16:53:54 ubuntu.local ticky: INFO Commented on ticket [#3813] (mcintosh)', 'Jan 31 16:54:18 ubuntu.local ticky: ERROR Connection to DB failed (bpacheco)', 'Jan 31 17:15:47 ubuntu.local ticky: ERROR The ticket was modified while updating (mcintosh)', 'Jan 31 17:29:11 ubuntu.local ticky: ERROR Connection to DB failed (oren)', 'Jan 31 17:51:52 ubuntu.local ticky: INFO Closed ticket [#8604] (mcintosh)', 'Jan 31 18:09:17 ubuntu.local ticky: ERROR The ticket was modified while updating (noel)', "Jan 31 18:43:01 ubuntu.local ticky: ERROR Ticket doesn't exist (nonummy)", 'Jan 31 19:00:23 ubuntu.local ticky: ERROR Timeout while retrieving information (blossom)', 'Jan 31 19:20:22 ubuntu.local ticky: ERROR Timeout while retrieving information (mai.hendrix)', 'Jan 31 19:59:06 ubuntu.local ticky: INFO Created ticket [#6361] (enim.non)', 'Jan 31 20:02:41 ubuntu.local ticky: ERROR Timeout while retrieving information (xlg)', 'Jan 31 20:21:55 ubuntu.local ticky: INFO Commented on ticket [#7159] (ahmed.miller)', 'Jan 31 20:28:26 ubuntu.local ticky: ERROR Connection to DB failed (breee)', 'Jan 31 20:35:17 ubuntu.local ticky: INFO Created ticket [#7737] (nonummy)', 'Jan 31 20:48:02 ubuntu.local ticky: ERROR Connection to DB failed (mdouglas)', 'Jan 31 20:56:58 ubuntu.local ticky: INFO Closed ticket [#4372] (oren)', 'Jan 31 21:00:23 ubuntu.local ticky: INFO Commented on ticket [#2389] (sri)', 'Jan 31 21:02:06 ubuntu.local ticky: ERROR Connection to DB failed (breee)', 'Jan 31 21:20:33 ubuntu.local ticky: INFO Closed ticket [#3297] (kirknixon)', 'Jan 31 21:29:24 ubuntu.local ticky: ERROR The ticket was modified while updating (blossom)', 'Jan 31 22:58:55 ubuntu.local ticky: INFO Created ticket [#2461] (jackowens)', 'Jan 31 23:25:18 ubuntu.local ticky: INFO Closed ticket [#9876] (blossom)', 'Jan 31 23:35:40 ubuntu.local ticky: INFO Created ticket [#5896] (mcintosh)']
#y is INFo and z is ERROR
y={}
z={}
a={}
b=[]
for i in range(len(x)):
    p=x[i]
    if(p.find("INFO")>0):
        t1=p.index('(')
        t2=p.index(')')
        t3=p[t1+1:t2]
        b.append(t3)
        if t3 in y.keys():
            y[t3]=y[t3]+1
        else:
            y[t3]=1
    else:
        t1=p.index('(')
        t2=p.index(')')
        t3=p[t1+1:t2]
        b.append(t3)
        if t3 in z.keys():
            z[t3]=z[t3]+1
        else:
            z[t3]=1
        t4=p.index('E')
        er=p[t4+6:t1-1]
        if er in a.keys():
            a[er]=a[er]+1
        else:
            a[er]=1
            
a=sorted(a.items(), key = operator.itemgetter(1), reverse=True)
b=set(b)
b=list(b)
b.sort()
with open('error_message.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Error", "Count"])
    for i in a:
        writer.writerow([i[0],i[1]])
    file.close()
with open('user_statistics.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Username", "INFO", "ERROR"])
    for i in range(len(b)):
        if not(b[i]) in y.keys():
            y[b[i]]=0
        if not(b[i]) in z.keys():
            z[b[i]]=0
        writer.writerow([b[i],y[b[i]],z[b[i]]])
    file.close()
